"""USD authoring only; usable without Kit for structural checks.

The installed native MJCF importer collapses co-located hinges into a D6 joint,
changing joint names and coordinate semantics. This explicit serial conversion
keeps the trained 23 hinge coordinates/order. Auxiliary links have small positive
inertia, recorded in the manifest; original bodies retain their native inertias.
"""
import numpy as np
from sim2sim_contract import triangles


def build(stage, manifest, grid, *, robot_path='/World/Robot',ground_path='/World/Ground',scene=True,ground=True):
    from pxr import UsdGeom,UsdPhysics,UsdShade,Gf,Sdf,Vt
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    def vec(v):return Gf.Vec3f(*map(float,v))
    def quat(v):return Gf.Quatf(float(v[0]),vec(v[1:]))
    def extra(prim,schema,attrs):
        prim.AddAppliedSchema(schema)
        for name,kind,value in attrs:prim.CreateAttribute(name,kind).Set(value)
    if scene:
        physics_scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
        physics_scene.CreateGravityDirectionAttr(vec([0,0,-1]));physics_scene.CreateGravityMagnitudeAttr(abs(manifest['gravity']))
        extra(physics_scene.GetPrim(),'PhysxSceneAPI',[
            ('physxScene:solverType',Sdf.ValueTypeNames.Token,'TGS'),
            ('physxScene:enableGPUDynamics',Sdf.ValueTypeNames.Bool,False),
            ('physxScene:broadphaseType',Sdf.ValueTypeNames.Token,'MBP')])
    UsdGeom.Xform.Define(stage,robot_path)
    bodies={b['id']:b for b in manifest['bodies']};paths={b['id']:robot_path+'/'+b['name'] for b in bodies.values()}
    def rigid(path,body,mass=None,inertia=None):
        x=UsdGeom.Xform.Define(stage,path)
        x.AddTranslateOp().Set(Gf.Vec3d(*body['pos']));x.AddOrientOp().Set(quat(body['quat']))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        ma=UsdPhysics.MassAPI.Apply(x.GetPrim());ma.CreateMassAttr(body['mass'] if mass is None else mass)
        ma.CreateCenterOfMassAttr(vec(body['com'] if mass is None else [0,0,0]))
        ma.CreateDiagonalInertiaAttr(vec(body['inertia'] if inertia is None else [inertia]*3))
        ma.CreatePrincipalAxesAttr(quat(body['inertia_quat'] if mass is None else [1,0,0,0]))
        extra(x.GetPrim(),'PhysxRigidBodyAPI',[
            ('physxRigidBody:linearDamping',Sdf.ValueTypeNames.Float,0.),
            ('physxRigidBody:angularDamping',Sdf.ValueTypeNames.Float,0.),
            ('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float,2.)])
    for b in bodies.values():rigid(paths[b['id']],b)
    root=stage.GetPrimAtPath(paths[1]);UsdPhysics.ArticulationRootAPI.Apply(root)
    extra(root,'PhysxArticulationAPI',[
        ('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,False),
        ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
        ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,8),
        ('physxArticulation:sleepThreshold',Sdf.ValueTypeNames.Float,0.),
        ('physxArticulation:stabilizationThreshold',Sdf.ValueTypeNames.Float,0.)])
    auxiliary=0
    for b in bodies.values():
        if b['parent']==0:continue
        previous=paths[b['parent']];previous_body=bodies[b['parent']]
        chain=b['joints'] or [None]
        for i,j in enumerate(chain):
            last=i==len(chain)-1
            target=paths[b['id']] if last else robot_path+'/aux_'+j['name']
            if not last:
                rigid(target,b,manifest['serial_auxiliary_mass_kg'],manifest['serial_auxiliary_inertia_kg_m2']);auxiliary+=1
            name=j['name'] if j else 'fixed_'+b['name']
            joint=(UsdPhysics.RevoluteJoint if j else UsdPhysics.FixedJoint).Define(stage,robot_path+'/Joints/'+name)
            joint.CreateBody0Rel().SetTargets([previous]);joint.CreateBody1Rel().SetTargets([target])
            # All exported local axes are native cardinal positive directions;
            # arbitrary body rest orientations remain encoded in joint frames.
            parentq=Gf.Quatd(previous_body['quat'][0],Gf.Vec3d(*previous_body['quat'][1:]))
            childq=Gf.Quatd(b['quat'][0],Gf.Vec3d(*b['quat'][1:]))
            axis=np.array(j['axis'] if j else [1,0,0],dtype=float)
            axisq=Gf.Rotation(Gf.Vec3d(1,0,0),Gf.Vec3d(*axis)).GetQuat()
            local0=parentq.GetInverse()*(childq*axisq)
            offset=Gf.Rotation(parentq.GetInverse()).TransformDir(Gf.Vec3d(*(np.array(b['pos'])-previous_body['pos'])))
            joint.CreateLocalPos0Attr(vec(offset));joint.CreateLocalPos1Attr(vec([0,0,0]))
            joint.CreateLocalRot0Attr(quat([local0.GetReal(),*local0.GetImaginary()]));joint.CreateLocalRot1Attr(quat([axisq.GetReal(),*axisq.GetImaginary()]))
            joint.CreateCollisionEnabledAttr(False)
            if j:
                joint.CreateAxisAttr('X');joint.CreateLowerLimitAttr(float(np.rad2deg(j['range'][0])));joint.CreateUpperLimitAttr(float(np.rad2deg(j['range'][1])))
                extra(joint.GetPrim(),'PhysxJointAPI',[('physxJoint:armature',Sdf.ValueTypeNames.Float,j['armature'])])
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
                drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.);drive.CreateMaxForceAttr(1e9)
            previous=target;previous_body=b
    material=UsdShade.Material.Define(stage,'/World/Looks/Contact')
    pm=UsdPhysics.MaterialAPI.Apply(material.GetPrim());pm.CreateStaticFrictionAttr(1.);pm.CreateDynamicFrictionAttr(1.);pm.CreateRestitutionAttr(0.)
    def collision(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        extra(prim,'PhysxCollisionAPI',[
            ('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.002),
            ('physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
    for g in manifest['geoms']:
        path=paths[g['body']]+'/'+g['name'];kind=g['type'];size=g['size']
        if kind==6:shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(2.);scale=size
        elif kind in (2,4):shape=UsdGeom.Sphere.Define(stage,path);shape.CreateRadiusAttr(1.);scale=[size[0]]*3 if kind==2 else size
        elif kind==3:
            shape=UsdGeom.Capsule.Define(stage,path);shape.CreateRadiusAttr(size[0]);shape.CreateHeightAttr(2*size[1]);shape.CreateAxisAttr('Z');scale=[1,1,1]
        else:raise ValueError(f'Unsupported native geom type {kind}')
        xf=UsdGeom.Xformable(shape);xf.AddTranslateOp().Set(Gf.Vec3d(*g['pos']));xf.AddOrientOp().Set(quat(g['quat']));xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        shape.CreateDisplayColorAttr([vec(g['rgba'][:3])])
        if g['collision']:collision(shape.GetPrim())
    topology={'root_path':paths[1],'body_paths':paths,'auxiliary_links':auxiliary,
              'extra_mass_kg':auxiliary*manifest['serial_auxiliary_mass_kg'],'ground_triangles':0}
    if not ground:return topology
    if grid is None:
        # USD plane-independent large flat mesh is bounded at 100m; episode
        # station/boundary protections stop the actor well before its edge.
        grid=np.zeros((2,2));half=50.
    else:half=manifest['terrain']['half_extent_m']
    points,faces=triangles(grid,half)
    ground=UsdGeom.Mesh.Define(stage,ground_path);ground.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    ground.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
    ground.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()));ground.CreateSubdivisionSchemeAttr('none')
    ground.CreateDisplayColorAttr([vec([.38,.37,.35])]);collision(ground.GetPrim())
    if manifest['terrain']['mode'] != 'flat':
        from sim2sim_visual import apply_lunar_appearance
        apply_lunar_appearance(stage,ground_path,points)
    UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr('none')
    topology['ground_triangles']=len(faces)
    return topology
