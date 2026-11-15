"""Display-only LOD regression with native static rays; no stepping or GL."""
from pathlib import Path
import sys
import unittest
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lunar_view_lod import install,DisplayLod


class LunarViewLod(unittest.TestCase):
    def test_near_surface_boundary_and_exclusive_far_surface(self):
        from scipy.spatial import Delaunay
        n=33;axis=np.linspace(-8,8,n);x,y=np.meshgrid(axis,axis)
        z=-.05*np.exp(-((x-.1)**2+(y-.2)**2)/.5)+.002*x
        vertices=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
        outer=np.array([[-20,-20,0],[20,-20,0],[20,20,0],[-20,20,0]])
        vertices=np.concatenate((vertices,outer))
        edge=(abs(x)==8)|(abs(y)==8)
        background_ids=np.r_[np.flatnonzero(edge),np.arange(n*n,n*n+4)]
        background=background_ids[Delaunay(vertices[background_ids,:2]).simplices]
        middle=vertices[background,:2].mean(1)
        background=background[~(abs(middle)<8-1e-8).all(-1)]
        a=(np.arange(n-1)[:,None]*n+np.arange(n-1)[None,:]).ravel()
        fine=np.concatenate((np.stack((a,a+1,a+n+1),-1),np.stack((a,a+n+1,a+n),-1)))
        faces=np.concatenate((background,fine))
        spec=mujoco.MjSpec();parent=spec.worldbody.add_body(name='site20_ground',mocap=True)
        spec.add_mesh(name='site20_full_view',uservert=vertices.ravel().tolist(),userface=faces.ravel().tolist(),maxhullvert=4)
        parent.add_geom(name='site20_full_visual',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='site20_full_view',
                        contype=0,conaffinity=0,mass=0,rgba=[.42,.43,.45,1])
        reference=spec.compile();data=mujoco.MjData(reference);mujoco.mj_kinematics(reference,data)
        meta=install(spec,{'normalized':z[None],'centers':np.array([[0,0]])})
        model=spec.compile();display=mujoco.MjData(model);mujoco.mj_kinematics(model,display)
        lod=DisplayLod(model,meta);lod.select([0])
        self.assertEqual(model.nq,reference.nq)
        np.testing.assert_array_equal(model.geom_contype,0);np.testing.assert_array_equal(model.geom_conaffinity,0)
        def ray(m,d,xy):
            distance=mujoco.mj_ray(m,d,np.r_[xy,3.],np.array([0.,0.,-1.]),None,True,-1,None)
            self.assertGreater(distance,0);return 3-distance
        for xy in ([.13,.17],[7.999,.553],[8.001,.553],[-7.999,-.431]):
            self.assertAlmostEqual(ray(model,display,xy),ray(reference,data,xy),delta=1e-5)
        # Boundaries use the identical source samples in BOTH LODs.
        fine_mesh=model.geom_dataid[lod.fine[0]];far_mesh=model.geom_dataid[lod.far[0]]
        def world_vertices(mid,gid):
            a,n=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
            return model.mesh_vert[a:a+n]@display.geom_xmat[gid].reshape(3,3).T+display.geom_xpos[gid]
        full=world_vertices(fine_mesh,lod.fine[0]);coarse=world_vertices(far_mesh,lod.far[0])
        boundary=full[(np.abs(np.abs(full[:,:2])-8)<1e-5).any(-1)]
        for point in boundary:self.assertLess(np.linalg.norm(coarse-point,axis=-1).min(),1e-5)
        original=lod.triangles();lod.select([])
        self.assertLess(lod.triangles(),original)
        self.assertEqual(float(model.geom_rgba[lod.fine[0],3]),0.)
        self.assertEqual(float(model.geom_rgba[lod.far[0],3]),1.)


if __name__=='__main__':unittest.main()
