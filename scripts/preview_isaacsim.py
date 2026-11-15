"""Isaac Sim lunar scene preview, using MJLab terrain and existing USD/PBR assets.

No SimulationContext, timeline playback, policy loading or training. Defaults
to an interactive window; --frames bounds a render-only snapshot session.
"""
import argparse
import os
from pathlib import Path


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/isaacsim_preview/terrain.npz')
    parser.add_argument('--quality',choices=('realtime','path'),default='realtime')
    parser.add_argument('--focus',choices=('terrain','astronaut'),default='terrain')
    parser.add_argument('--no-astronaut',action='store_true')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--warmup',type=int,default=120)
    parser.add_argument('--spp',type=int,default=4)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--export-usd',type=Path)
    args=parser.parse_args()
    if args.frames<0 or args.warmup<1 or not 1<=args.spp<=32:parser.error('Invalid frames/warmup/spp')
    if args.headless and not args.frames:parser.error('--headless requires a finite --frames')
    if args.frames and args.snapshot and args.frames<args.warmup+10:parser.error('Snapshot requires frames >= warmup + 10')
    return args


def main():
    args=arguments()  # --help must work without importing/starting Kit.
    assets=Path(os.environ.get('ZBOT_MOON_DIR',str(Path.home()/'myWorks/zbot_exoarm_moon')))
    texture=assets/'assets/lunar_sites/southpole_site20/LunarRegolith4k'
    suit=assets/'outputs/wangyu_new_preview/wangyu_rigged.usdc'
    required=[args.terrain]+[texture/f'LunarRegolith4k_{name}.jpg' for name in ('diff','rough','nor_gl','ao')]
    if not args.no_astronaut:required.append(suit)
    for path in required:
        if not path.is_file():raise FileNotFoundError(f'Required local scene resource missing: {path}')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':args.headless,'width':1280,'height':720,'limit_cpu_threads':4,
                       'renderer':'PathTracing' if args.quality=='path' else 'RayTracedLighting'})
    try:
        import carb
        import numpy as np
        import omni.usd
        import omni.timeline
        from pxr import UsdGeom,UsdShade,UsdLux,Gf,Sdf,Vt
        from isaacsim.core.utils.viewports import set_camera_view
        settings=carb.settings.get_settings()
        settings.set('/app/viewport/grid/enabled',False)
        settings.set('/rtx/rendermode','PathTracing' if args.quality=='path' else 'RayTracedLighting')
        settings.set('/rtx/post/tonemap/op',4);settings.set('/rtx/post/tonemap/filmIso',100.)
        settings.set('/rtx/post/lensFlares/enabled',False);settings.set('/rtx/post/bloom/enabled',False)
        if args.quality=='path':
            settings.set('/rtx/pathtracing/spp',args.spp)
            settings.set('/rtx/pathtracing/totalSpp',args.spp*args.warmup)
            settings.set('/rtx/pathtracing/maxBounces',6)
            settings.set('/rtx/resetPtAccumOnAnimTimeChange',False)
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
        material=UsdShade.Material.Define(stage,'/World/Looks/Regolith')
        shader=UsdShade.Shader.Define(stage,'/World/Looks/Regolith/Shader')
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'),'mdl');shader.SetSourceAssetSubIdentifier('OmniPBR','mdl')
        shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
        for name,suffix in (('diffuse_texture','diff'),('normalmap_texture','nor_gl'),
                            ('reflectionroughness_texture','rough'),('ao_texture','ao')):
            shader.CreateInput(name,Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture/f'LunarRegolith4k_{suffix}.jpg')))
        for name,value in (('metallic_constant',0.),('reflection_roughness_constant',1.),
                           ('reflection_roughness_texture_influence',.13),('ao_to_diffuse',.35),
                           ('specular_level',.26),('bump_factor',.8)):
            shader.CreateInput(name,Sdf.ValueTypeNames.Float).Set(value)
        shader.CreateInput('flip_tangent_v',Sdf.ValueTypeNames.Bool).Set(True)
        material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
        with np.load(args.terrain,allow_pickle=False) as archive:
            import json
            meta=json.loads(str(archive['metadata']))
            for index,name in enumerate(meta['meshes']):
                mesh=UsdGeom.Mesh.Define(stage,f'/World/Terrain/{name}')
                points=archive[f'points_{index}'];faces=archive[f'faces_{index}']
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
                mesh.CreateSubdivisionSchemeAttr('none')
                uv=(points[:,:2]/4).astype(np.float32)
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv))
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        sun=UsdLux.DistantLight.Define(stage,'/World/Sun')
        sun.CreateIntensityAttr(1750.);sun.CreateAngleAttr(.53);sun.CreateColorAttr(Gf.Vec3f(1,1,1))
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(0,45,180))
        if not args.no_astronaut:
            actor=UsdGeom.Xform.Define(stage,'/World/Wangyu')
            actor.GetPrim().GetReferences().AddReference(str(suit),'/root')
            bbox=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render]).ComputeWorldBound(actor.GetPrim()).ComputeAlignedRange()
            lo,hi=bbox.GetMin(),bbox.GetMax()
            if bbox.IsEmpty():raise RuntimeError('Wangyu USD reference has no visible bounds')
            actor.AddTranslateOp().Set(Gf.Vec3d(2-float((lo[0]+hi[0])/2),-float((lo[1]+hi[1])/2),
                                               meta['ground_heights']['actor']+.02-float(lo[2])))
        ground=meta['ground_heights']
        if args.focus=='terrain':eye,target=[0,-5,ground['terrain_eye']+1.1],[0,5,ground['terrain_target']+.5]
        else:eye,target=[4,-3,ground['actor_eye']+1.7],[2,0,ground['actor']+1.]
        set_camera_view(eye=eye,target=target,camera_prim_path='/OmniverseKit_Persp')
        omni.timeline.get_timeline_interface().stop()
        print(f'[Isaac scene] static preview | {meta["physics_revision"]} | {meta["terrain_fingerprint"]}',flush=True)
        print(f'[Isaac assets] {suit if not args.no_astronaut else "no astronaut"} | regolith={texture}',flush=True)
        if args.export_usd:
            args.export_usd.parent.mkdir(parents=True,exist_ok=True)
            if not stage.GetRootLayer().Export(str(args.export_usd.resolve())):raise RuntimeError('USD export failed')
        captures=[];frame=0
        while app.is_running() and (not args.frames or frame<args.frames):
            app.update();frame+=1
            if args.snapshot and frame==args.warmup:
                from omni.kit.viewport.utility import capture_viewport_to_file,get_active_viewport
                args.snapshot.parent.mkdir(parents=True,exist_ok=True)
                captures.append(capture_viewport_to_file(get_active_viewport(),str(args.snapshot.resolve())))
    finally:app.close()


if __name__=='__main__':main()
