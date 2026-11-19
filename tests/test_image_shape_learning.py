"""Shape features must ignore camera framing and reject invalid hand data."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from image_shape_learning import shape_features, ImageShapeClassifier


class ShapeTest(unittest.TestCase):
    def test_invariant_to_mirror_rotation_scale_and_translation(self):
        points=np.random.default_rng(42).uniform(.1,.8,(21,3))
        base=shape_features(points,640,480)
        xy=points[:,:2]*[1.,.75]
        angle=.7
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        xy=xy@rotation*1.8+[.2,-.1]
        xy[:,0]=-xy[:,0]
        transformed=np.column_stack([xy[:,0],xy[:,1]/.75,points[:,2]])
        np.testing.assert_allclose(base,shape_features(transformed,640,480),atol=1e-12)

    def test_rejects_degenerate_nonfinite_or_invalid_points(self):
        for points in (np.zeros((21,3)),np.ones((20,3)),np.full((21,3),np.nan)):
            with self.assertRaises(ValueError): shape_features(points)

    def test_learned_rest_is_separate_from_uncertain_prediction(self):
        classifier=ImageShapeClassifier()
        model=Mock(); model.classes_=np.array(['PINCH_PINKY','PINCH_RING','PINCH_THUMB','UNKNOWN'])
        classifier.artifact={'model':model,'threshold':.6,'report':{'model':'test'}}
        points=np.random.default_rng(42).uniform(.1,.8,(21,3))
        model.predict_proba.return_value=np.array([[.02,.03,.05,.9]])
        self.assertEqual(classifier.predict(points,640,480)['pose'],'IDLE')
        model.predict_proba.return_value=np.array([[.2,.3,.2,.3]])
        result=classifier.predict(points,640,480)
        self.assertEqual(result['pose'],'UNKNOWN')
        self.assertFalse(result['accepted'])


if __name__=='__main__': unittest.main()
