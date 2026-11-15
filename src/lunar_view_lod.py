"""CPU viewer-only near/far terrain assets. Training/collision assets stay intact.

The standalone full welded mesh remains the export/physical audit reference.
Split its exact background and fine faces; only inactive patches use a 1m
interior display LOD. Every .1m boundary vertex is kept at its source height.
"""
import numpy as np
import mujoco


def install(spec,bank):
    from scipy.spatial import Delaunay
    geom=next(g for g in spec.geoms if g.name=='site20_full_visual')
    mesh=next(m for m in spec.meshes if m.name==geom.meshname)
    vertices=np.asarray(mesh.uservert,dtype=np.float32).reshape(-1,3)
    faces=np.asarray(mesh.userface,dtype=np.int32).reshape(-1,3)
    n=bank['normalized'].shape[1];count=len(bank['centers']);per_patch=2*(n-1)**2
    background=len(faces)-count*per_patch
    if background<=0:raise ValueError('Unsupported full terrain face layout')
    parent=next(b for b in spec.bodies if b.name=='site20_ground')
    pose=np.array(geom.pos);rotation=np.array(geom.quat);rgba=np.array(geom.rgba);group=geom.group
    spec.delete(geom);spec.delete(mesh)
    def add(name,points,triangles,visible):
        spec.add_mesh(name=name,uservert=points.ravel().tolist(),userface=triangles.ravel().tolist(),
                      maxhullvert=4,smoothnormal=False)
        color=rgba.copy();color[3]=float(visible)
        parent.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,
                        pos=pose,quat=rotation,rgba=color,group=group,mass=0,contype=0,conaffinity=0)
    def compact(triangles):
        used,inverse=np.unique(triangles,return_inverse=True)
        return vertices[used],inverse.reshape(-1,3).astype(np.int32)
    points,triangles=compact(faces[:background]);add('site20_view_background',points,triangles,True)
    fine_names=[];coarse_names=[];far_counts=[]
    for index,center in enumerate(bank['centers']):
        triangles=faces[background+index*per_patch:background+(index+1)*per_patch]
        points,local_faces=compact(triangles)
        # Source grid nodes are used verbatim, including all fine boundary
        # nodes shared with the background. No overlapping coarse top is added.
        half=8.;step=2*half/(n-1)
        ij=np.rint((points[:,:2]-center+half)/step).astype(int)
        if not ((ij>=0)&(ij<n)).all():raise ValueError('Fine patch faces do not match their source center')
        edge=((ij==0)|(ij==n-1)).any(-1)
        stride=max(1,round(1./step))
        keep=edge|((ij%stride)==0).all(-1)
        coarse=points[keep];coarse_faces=Delaunay(coarse[:,:2]).simplices.astype(np.int32)
        fine=f'site20_view_fine_{index:02d}';far=f'site20_view_far_{index:02d}'
        add(fine,points,local_faces,False);add(far,coarse,coarse_faces,True)
        fine_names.append(fine);coarse_names.append(far);far_counts.append(len(coarse_faces))
    return {'fine_names':fine_names,'far_names':coarse_names,'far_triangles':far_counts,
            'background_triangles':background,'fine_triangles_per_patch':per_patch,
            'original_triangles':len(faces),'fine_resolution_m':2*half/(n-1),'far_interior_resolution_m':1.}


class DisplayLod:
    def __init__(self,model,metadata):
        self.model=model;self.metadata=metadata
        self.fine=np.array([model.geom(name).id for name in metadata['fine_names']])
        self.far=np.array([model.geom(name).id for name in metadata['far_names']])
        self.active=None

    def select(self,patches):
        active=np.zeros(len(self.fine),dtype=bool);active[np.asarray(patches,dtype=int)]=True
        if self.active is not None and np.array_equal(active,self.active):return
        self.model.geom_rgba[self.fine,3]=active
        self.model.geom_rgba[self.far,3]=~active
        self.active=active

    def triangles(self):
        active=np.zeros(len(self.fine),dtype=bool) if self.active is None else self.active
        return int(self.metadata['background_triangles']+active.sum()*self.metadata['fine_triangles_per_patch']+
                   np.array(self.metadata['far_triangles'])[~active].sum())
