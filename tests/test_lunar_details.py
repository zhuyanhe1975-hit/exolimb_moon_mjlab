"""Procedural relief invariants; no simulation."""
import unittest
from dataclasses import replace
import numpy as np
from lunar_details import generate,DetailConfig


class LunarDetails(unittest.TestCase):
    def test_repeatable_global_coordinates_and_overlapping_crops(self):
        axis=np.linspace(-8,8,161);x,y=np.meshgrid(axis,axis);xy=np.stack((x,y),-1)
        cfg=DetailConfig();a=generate(xy,[25,40],cfg)
        np.testing.assert_array_equal(a,generate(xy,[25,40],cfg))
        self.assertFalse(np.array_equal(a,generate(xy,[30,40],cfg)))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(a.max()),.003)
        self.assertLess(float(a.min()),-.01)
        self.assertLess(float(abs(a).max()),.15)
        # Same world points from different crop centers and request sizes.
        points=xy[::13,::11]+[25,40]
        expected=generate(points-[25,40],[25,40],cfg)
        np.testing.assert_allclose(expected,generate(points-[30,35],[30,35],cfg),atol=1e-11,rtol=0)
        for point,value in zip(points.reshape(-1,2)[::9],expected.ravel()[::9]):
            self.assertAlmostEqual(float(generate(point[None,:],[0,0],cfg)[0]),float(value),places=10)
        np.testing.assert_array_equal(generate(xy,[25,40],replace(cfg,enabled=False)),0.)

    def test_small_features_have_resolvable_widths(self):
        cfg=DetailConfig()
        sigma=max(.16*min(cfg.crater_radius_m),cfg.min_rim_sigma_cells*cfg.resolution_m)
        self.assertGreaterEqual(2.355*sigma/cfg.resolution_m,3.5)
        self.assertGreaterEqual(2*.8*min(cfg.mound_radius_m)/cfg.resolution_m,6)
        with self.assertRaises(ValueError):replace(cfg,min_rim_sigma_cells=.5)

    def test_sparse_world_queries_match_individual_queries(self):
        points=np.array([[-951.,-799.],[801.,624.],[25.,40.]])
        expected=np.array([generate(p[None,:],[0,0])[0] for p in points])
        np.testing.assert_allclose(generate(points,[0,0]),expected,atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
