"""Shared visual settings matching the project's Isaac lunar scene preview."""
import os
from pathlib import Path
import numpy as np


def apply_lunar_appearance(stage, ground_path, points):
    from pxr import UsdGeom, UsdShade, Sdf, Vt
    root=Path(os.environ.get('ZBOT_MOON_DIR',str(Path.home()/'myWorks/zbot_exoarm_moon')))
    texture=root/'assets/lunar_sites/southpole_site20/LunarRegolith4k'
    files={suffix:texture/f'LunarRegolith4k_{suffix}.jpg' for suffix in ('diff','nor_gl','rough','ao')}
    for path in files.values():
        if not path.is_file():raise FileNotFoundError(f'Required lunar scene texture missing: {path}')
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(ground_path))
    if not mesh:raise ValueError(f'Lunar ground mesh missing: {ground_path}')
    uv=(np.asarray(points,dtype=np.float32)[:,:2]/4).astype(np.float32)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv))
    material=UsdShade.Material.Define(stage,'/World/Looks/Regolith')
    shader=UsdShade.Shader.Define(stage,'/World/Looks/Regolith/Shader')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'),'mdl');shader.SetSourceAssetSubIdentifier('OmniPBR','mdl')
    shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
    for name,suffix in (('diffuse_texture','diff'),('normalmap_texture','nor_gl'),('reflectionroughness_texture','rough'),('ao_texture','ao')):
        shader.CreateInput(name,Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(files[suffix])))
    for name,value in (('metallic_constant',0.),('reflection_roughness_constant',1.),('reflection_roughness_texture_influence',.13),('ao_to_diffuse',.35),('specular_level',.26),('bump_factor',.8)):
        shader.CreateInput(name,Sdf.ValueTypeNames.Float).Set(value)
    shader.CreateInput('flip_tangent_v',Sdf.ValueTypeNames.Bool).Set(True)
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return texture


def apply_preview_lighting(stage):
    import carb
    from pxr import UsdGeom,UsdLux,Gf
    settings=carb.settings.get_settings()
    settings.set('/app/viewport/grid/enabled',False)
    settings.set('/rtx/rendermode','RayTracedLighting')
    settings.set('/rtx/post/tonemap/op',4);settings.set('/rtx/post/tonemap/filmIso',100.)
    settings.set('/rtx/post/lensFlares/enabled',False);settings.set('/rtx/post/bloom/enabled',False)
    sun=UsdLux.DistantLight.Define(stage,'/World/Sun')
    sun.CreateIntensityAttr(1750.);sun.CreateAngleAttr(.53);sun.CreateColorAttr(Gf.Vec3f(1,1,1))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(0,45,180))
