"""Display native MuJoCo-deformed suit vertices in USD; no physics influence."""
import os
from pathlib import Path
import mujoco
import numpy as np


class SuitDisplay:
    def __init__(self, stage, sensors, *, robot_path='/World/Robot',suit_path='/World/Suit',offset=None):
        from pxr import UsdGeom,UsdShade,Sdf,Gf,Vt
        self.sensors=sensors;self.meshes=[];self.attributes=[];m=sensors.model
        parent=UsdGeom.Xform.Define(stage,suit_path)
        if offset is not None:parent.AddTranslateOp().Set(Gf.Vec3d(*map(float,offset)))
        self.scene=mujoco.MjvScene(m,maxgeom=256)
        self.option=mujoco.MjvOption();self.camera=mujoco.MjvCamera()
        self.refresh()
        # Reference the very same material graphs used by run_isaacsim.sh.
        # References retain their source-layer-relative texture resolution.
        source=Path(os.environ.get('ZBOT_MOON_DIR',str(Path.home()/'myWorks/zbot_exoarm_moon')))/'outputs/wangyu_new_preview/wangyu_rigged.usdc'
        if not source.is_file():raise FileNotFoundError(f'Preview suit material source missing: {source}')
        material_names={0:'Helmet_MAT_VRay',1:'Helmet_MAT_VRay',2:'Boot_MAT_VRay',3:'Suit_MAT_VRay'}
        materials={}
        for matid in range(m.nmat):
            if matid not in material_names:raise ValueError(f'Unexpected Wangyu skin material index {matid}')
            name=material_names[matid]
            path=f'/World/Looks/{name}'
            material=UsdShade.Material.Define(stage,path)
            if not material.GetPrim().GetMetadata('references'):
                material.GetPrim().GetReferences().AddReference(str(source),f'/root/_materials/{name}')
            if not material.ComputeSurfaceSource()[0]:raise RuntimeError(f'Preview material surface missing: {name}')
            materials[matid]=material
        for i in range(m.nskin):
            mesh=UsdGeom.Mesh.Define(stage,f'{suit_path}/Skin{i}')
            a,n=int(m.skin_faceadr[i]),int(m.skin_facenum[i]);faces=m.skin_face[a:a+n]
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n,3,dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            a,n=int(m.skin_texcoordadr[i]),int(m.skin_vertnum[i])
            if a>=0:
                uv=m.skin_texcoord[a:a+n].copy()
                # USD textures use bottom-origin UV, whereas MuJoCo's native
                # texture loader uses the vflip=true source convention here.
                # Preserve the source Blender UV, with no second image flip.
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[int(m.skin_matid[i])])
            self.meshes.append(mesh)
            mesh.SetNormalsInterpolation('vertex')
            self.attributes.append((mesh.GetPointsAttr(),mesh.CreateNormalsAttr()))
        # Hide only source debug shapes. Rigid bodies and shoe collisions stay active.
        for geom in sensors.manifest['geoms']:
            body=sensors.manifest['bodies'][geom['body']-1]['name']
            UsdGeom.Imageable(stage.GetPrimAtPath(f'{robot_path}/{body}/{geom["name"]}')).MakeInvisible()
        self.update()

    def update(self):
        update_suits([self])

    def prepare_update(self):
        from pxr import Vt
        m=self.sensors.model
        self.refresh()
        updates=[]
        for i,(points,normals) in enumerate(self.attributes):
            a,n=int(m.skin_vertadr[i]),int(m.skin_vertnum[i])
            updates.append((points,Vt.Vec3fArray.FromNumpy(self.scene.skinvert.reshape(-1,3)[a:a+n].copy())))
            updates.append((normals,Vt.Vec3fArray.FromNumpy(self.scene.skinnormal.reshape(-1,3)[a:a+n].copy())))
        return updates

    def refresh(self):
        # 3.12 deprecates mjv_updateSkin but does not expose its replacement in
        # Python. The supported native scene update deforms active skins too.
        # This allocates no renderer and advances no physics.
        mujoco.mjv_updateScene(self.sensors.model,self.sensors.data,self.option,None,
                              self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)


def update_suits(suits):
    """Compute independent native skins, then coalesce USD change notices."""
    from pxr import Sdf
    updates=[item for suit in suits for item in suit.prepare_update()]
    with Sdf.ChangeBlock():
        for attr,value in updates:attr.Set(value)
