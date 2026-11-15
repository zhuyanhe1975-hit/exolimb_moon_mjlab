"""Projected native-contact geometry + LIPM capture proxy, not a wrench-feasibility proof.

No nominal sole corners are invented. Degenerate point/line support has no positive
interior margin. Uneven terrain, angular momentum and external wrenches violate
LIPM assumptions: retain the name 'proxy', not ZMP or a general viability margin.
"""
import os
from functools import partial
import torch
from tensor_graph import FixedTensorGraph
import balance_mdp as base

OBS_DIM=151


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def signed_margin(points,valid,query):
    """Batched signed Euclidean distance to convex hull, including degeneracies.

    positive inside a 2D hull, negative outside; zero-width hull never positive.
    Small fixed contact sets only (8 slots). All operations remain on device.
    """
    delta=points[:,None,:,:]-points[:,:,None,:]  # i -> j
    length2=delta.square().sum(-1)
    pair=valid[:,:,None]&valid[:,None,:]&(length2>1e-10)
    rel=points[:,None,None,:,:]-points[:,:,None,None,:]
    side=cross2(delta[:,:,:,None,:],rel)
    edge=pair&((side>=-1e-7)|~valid[:,None,None,:]).all(-1)
    area=(side.abs().masked_fill(~(pair[:,:,:,None]&valid[:,None,None,:]),0).amax((1,2,3))>1e-7)
    qrel=query[:,None,None,:]-points[:,:,None,:]
    fraction=(qrel*delta).sum(-1)/length2.clamp_min(1e-10)
    distance=(qrel-fraction.clamp(0,1)[...,None]*delta).norm(dim=-1)
    eligible=torch.where(area[:,None,None],edge,pair)
    closest=distance.masked_fill(~eligible,float('inf')).amin((1,2))
    single=(query[:,None,:]-points).norm(dim=-1).masked_fill(~valid,float('inf')).amin(-1)
    closest=torch.where(pair.any((1,2)),closest,single)
    inside=((cross2(delta,qrel)>=-1e-7)|~edge).all((1,2))&area
    margin=torch.where(inside,closest,-closest)
    margin=torch.where(valid.any(-1),margin,torch.full_like(margin,-.2))
    return margin,area


def compute_geometry(points,valid,com,velocity,gravity):
    count=valid.sum(-1)
    center=(points*valid[:,:,None]).sum(1)/count.clamp_min(1)[:,None]
    # Center first: 2km map coordinates must not enter hull cross products.
    local=points[:,:,:2]-center[:,None,:2]
    relative=com[:,:2]-center[:,:2]
    height=(com[:,2]-center[:,2]).clamp(.2,1.5)
    capture=relative+(height/gravity).sqrt()[:,None]*velocity[:,:2]
    static,area=signed_margin(local,valid,relative)
    dynamic,_=signed_margin(local,valid,capture)
    # No contact: an explicitly invalid proxy, never an accidental healthy value.
    capture=torch.where((count>0)[:,None],capture,torch.zeros_like(capture))
    return torch.cat((static[:,None],dynamic[:,None],capture,
                      count[:,None].float()/points.shape[1],area[:,None].float()),-1)


def geometry_input(x,gravity):
    contacts=x[:,:32].reshape(-1,8,4)
    geometry=compute_geometry(contacts[:,:,:3],contacts[:,:,3]>.5,x[:,32:35],x[:,35:38],gravity)
    w,a,b,z=x[:,38:42].unbind(-1)
    cosine=1-2*(b*b+z*z);sine=2*(w*z+a*b)
    scale=(cosine.square()+sine.square()).sqrt().clamp_min(1e-6)
    cosine=cosine/scale;sine=sine/scale
    vx,vy=geometry[:,2],geometry[:,3]
    local=torch.stack((cosine*vx+sine*vy,-sine*vx+cosine*vy),-1)
    return torch.cat((geometry[:,:2],local,geometry[:,4:]),-1)


def phase_contact_data(env,phase):
    """Refresh the public sensor cache at the post-forward observation boundary.

    MJLab updates sensors in the physics loop, but does not invalidate their
    concatenated ContactData after the final sim.forward/sense if no reset took
    place. dt=0 invalidates that cache without advancing physics or sensor time.
    This sensor must remain stateless: history tracking would push an extra frame.
    """
    if phase not in ('reward','observation'):
        raise ValueError(f'Unknown support data phase: {phase}')
    sensor=env.scene['support_contacts']
    if phase=='observation':
        if sensor.cfg.track_air_time or sensor.cfg.history_length:
            raise ValueError('Support observation cache refresh requires a stateless ContactSensor')
        sensor.update(dt=0.)
    return sensor.data


def data(env,phase='reward'):
    key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0),phase)
    if getattr(env,'_support_key',None)!=key:
        base.data(env,phase)
        contact=phase_contact_data(env,phase)
        points=contact.pos
        valid=(contact.found>0)&(contact.force.norm(dim=-1)>.5)&(contact.dist<=.002)
        valid=valid&torch.isfinite(points).all(-1)
        points=torch.where(valid[:,:,None],points,torch.zeros_like(points))
        if not hasattr(env,'_support_com_indices'):
            model=env.sim.mj_model
            ids=[]
            for name,width in (('com_pos',3),('com_vel',3),('root_quat',4)):
                adr=int(model.sensor('robot/balance_'+name).adr[0]);ids.extend(range(adr,adr+width))
            env._support_com_indices=torch.tensor(ids,device=env.device)
        native=env.sim.data.sensordata[:,env._support_com_indices]
        packed=torch.cat((torch.cat((points,valid[:,:,None].float()),-1).flatten(1),native),-1)
        fn=partial(geometry_input,gravity=abs(env.cfg.sim.mujoco.gravity[2]))
        if packed.is_cuda and os.environ.get('WANGYU_TENSOR_GRAPH','1')!='0':
            if not hasattr(env,'_support_graph'):
                env._support_graph=FixedTensorGraph(fn,packed)
            env._support_values=env._support_graph(packed)
        else:
            env._support_values=fn(packed)
        env._support_key=key
    return env._support_values


def observation(env):
    geometry=data(env,'observation')
    scaled=torch.cat((geometry[:,:4]/.2,geometry[:,4:]),-1).clamp(-5,5)
    return torch.cat((base.observation(env),scaled),-1)


def capture_risk(env):
    # Encourage a 2cm interior buffer, without fixing a preferred body pose.
    return ((.02-data(env)[:,1]).clamp_min(0)/.1).square().clamp_max(4.)


def airborne(env):
    return (data(env)[:,4]==0).float()


def metric(env,column):
    return data(env)[:,column]


def torso_fallen(env):
    return base.signal(env,'torso_tilt_deg')>60.
