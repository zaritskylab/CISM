"""Known signal-location challenges, not a restatement of implementation details."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from analysis.marker_ownership import motif_marker_ownership as analysis


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((50,55),np.int32)
        self.mask[12:37,10:30]=1
        self.mask[12:37,30:44]=2
        self.own=self.mask==1
        self.boundary=self.own&~ndi.binary_erosion(self.own,structure=np.ones((3,3),bool),iterations=2)

    def measure(self,image,mask=None):
        return analysis.boundary_measurements(image,self.mask if mask is None else mask,1,2,1.,.01)

    def test_genuine_membrane_is_retained_away_from_bright_neighbor(self):
        image=np.zeros(self.mask.shape);image[self.boundary]=6;image[self.mask==2]=30
        result=self.measure(image)
        self.assertAlmostEqual(result['away_boundary_mean'],6)
        self.assertAlmostEqual(result['selective_interface_log2_loss'],0)
        self.assertEqual(result['away_positive_fraction'],1)

    def test_interface_patch_has_selective_loss(self):
        image=np.zeros(self.mask.shape);image[self.mask==2]=30
        image[self.own&(np.indices(self.mask.shape)[1]>=27)]=6
        result=self.measure(image)
        self.assertGreater(result['selective_interface_log2_loss'],2)
        self.assertEqual(result['away_positive_pixels'],0)
        self.assertEqual(result['away_boundary_mean'],0)

    def test_equal_whole_mean_different_spatial_coverage(self):
        membrane=np.zeros(self.mask.shape);membrane[self.boundary]=6
        patch=np.zeros(self.mask.shape);location=self.boundary&(np.indices(self.mask.shape)[1]>=27)
        patch[location]=membrane[self.own].sum()/location.sum()
        self.assertAlmostEqual(patch[self.own].mean(),membrane[self.own].mean())
        self.assertGreater(self.measure(membrane)['away_positive_fraction'],self.measure(patch)['away_positive_fraction'])

    def test_neighbor_signal_alone_is_not_intrinsic_signal(self):
        image=np.zeros(self.mask.shape);image[self.mask==2]=100
        result=self.measure(image)
        self.assertEqual(result['interface_signal_sum'],0)
        self.assertEqual(result['away_positive_fraction'],0)
        self.assertAlmostEqual(result['selective_interface_log2_loss'],0)

    def test_area_normalization_preserves_uniform_cytoplasmic_mean(self):
        image=np.zeros(self.mask.shape);image[self.own]=8;image[self.mask==2]=40
        result=self.measure(image)
        self.assertAlmostEqual(result['retained_cell_mean'],8)
        self.assertAlmostEqual(result['sham_cell_retained_mean'],8)

    def test_missing_mask_is_unmeasured(self):
        result=self.measure(np.zeros(self.mask.shape),np.zeros(self.mask.shape,np.int32))
        self.assertEqual(result['boundary_failure'],'missing_mask')
        self.assertNotIn('selective_interface_log2_loss',result)

    def test_inadequate_sham_area_is_not_a_zero_result(self):
        mask=np.zeros((15,15),np.int32);mask[4:8,4:8]=1;mask[3:9,8:10]=2
        image=np.ones(mask.shape)
        result=analysis.boundary_measurements(image,mask,1,2,.5,.01)
        self.assertEqual(result['boundary_failure'],'insufficient_interface_or_sham_area')
        self.assertTrue(np.isnan(result['selective_interface_log2_loss']))


class CalibrationTests(unittest.TestCase):
    def example(self,n_controls=60,selective_loss=2.):
        rows=[]
        for i in range(n_controls+11):
            role='matched_unidentified_control' if i<n_controls else 'annotated_reference' if i<n_controls+10 else 'candidate'
            rows.append(dict(cell_id=i+1,marker='MECA-79',role=role,pred_mapped='hevs' if role=='annotated_reference' else 'unidentified',
                geometry_valid=True,whole_mean=5. if role!='matched_unidentified_control' else .2,
                donor_mean=1.,pseudocount=.01,negative_reference_mean_q95=.1,
                mask_area=200+i%3,crowding_50px=12+i%4,away_boundary_mean=6.,away_positive_fraction=.5,
                retained_cell_mean=5.,away_positive_pixels=10,removed_interface_pixels=10,
                interface_signal_sum=10.,interface_log2_asymmetry=.1,donor_log2_ratio=.1,
                selective_interface_log2_loss=.1))
        rows[-1].update(interface_log2_asymmetry=3.,donor_log2_ratio=3.,selective_interface_log2_loss=selective_loss)
        return pd.DataFrame(rows)

    def test_consistent_interface_and_source_pattern_can_flag(self):
        d=analysis.calibrate(self.example())
        self.assertEqual(d.iloc[-1].ownership_risk_q95,1)
        self.assertGreaterEqual(d.iloc[-1].risk_reference_n,30)

    def test_bright_neighbor_without_selective_loss_does_not_flag(self):
        d=analysis.calibrate(self.example(selective_loss=0.))
        self.assertEqual(d.iloc[-1].ownership_risk_q95,0)
        self.assertTrue(d.iloc[-1].away_marker_supported)

    def test_too_few_controls_is_missing_not_negative(self):
        d=analysis.calibrate(self.example(n_controls=29))
        self.assertTrue(np.isnan(d.iloc[-1].ownership_risk_q95))

    def test_no_interface_cannot_flag(self):
        d=self.example();d.loc[d.index[-1],'removed_interface_pixels']=0
        d.loc[d.index[-1],['interface_log2_asymmetry','donor_log2_ratio','selective_interface_log2_loss']]=np.nan
        result=analysis.calibrate(d)
        self.assertEqual(result.iloc[-1].ownership_risk_q95,0)


if __name__=='__main__':unittest.main()
