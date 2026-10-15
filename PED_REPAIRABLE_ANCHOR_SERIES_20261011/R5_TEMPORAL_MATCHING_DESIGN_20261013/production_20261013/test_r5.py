import unittest,inspect
import numpy as np
import features_r5 as f,core_r5 as k
def guard():pass
class Tests(unittest.TestCase):
 def example(self):
  p=np.arange(16,dtype=float).reshape(1,8,2)/16;p-=p[:,-1:];n=np.arange(480,dtype=float).reshape(1,20,12,2)/32
  return p,n
 def test_shape_order_scalar(self):
  p,n=self.example();a=f.build(p.astype("f4"),n.astype("f4"));b=f.independent(p.astype("f4"),n.astype("f4"));np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(1,20,168))
 def test_stationary(self):
  _,n=self.example();np.testing.assert_array_equal(f.build(np.zeros((1,8,2)),n),np.zeros((1,20,168)))
 def test_CV_and_first_connection(self):
  p=np.zeros((1,8,2));p[0,:,0]=np.arange(-7,1)*.4;n=np.zeros((1,20,12,2));n[:,:,:,0]=np.arange(1,13)*.4
  a=f.primitive(p,n);np.testing.assert_allclose(a[:,:,::2],1,rtol=0,atol=1e-12);np.testing.assert_array_equal(a[:,:,1::2],0)
 def test_rotation_reflection(self):
  p,n=self.example();theta=.7;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);a=f.primitive(p,n);b=f.primitive(p@rot,n@rot)
  np.testing.assert_allclose(a,b,rtol=0,atol=1e-12);ref=f.primitive(p*np.array([1,-1]),n*np.array([1,-1]));np.testing.assert_allclose(a[:,:,::2],ref[:,:,::2],rtol=0,atol=1e-12);np.testing.assert_allclose(a[:,:,1::2],-ref[:,:,1::2],rtol=0,atol=1e-12)
 def test_accel_decel_turn(self):
  p,n=self.example();p[:,:,0]=np.arange(8)**2/8;p-=p[:,-1:];n[:,:,0,1]=-.3;n[:,:,1,1]=.3
  np.testing.assert_array_equal(f.build(p.astype("f4"),n.astype("f4")),f.independent(p.astype("f4"),n.astype("f4")))
 def test_stats_train_only(self):
  T=np.arange(3*20*168,dtype="f4").reshape(3,20,168);s=f.statistics(T,np.array([True,True,False]));T[2]+=1e6;s2=f.statistics(T,np.array([True,True,False]))
  for key in s:np.testing.assert_array_equal(s[key],s2[key])
 def test_no_GT_metadata_signature(self):
  self.assertEqual(list(inspect.signature(f.build).parameters),["past","native"]);self.assertEqual(list(inspect.signature(k.score).parameters),["z","model"])
 def test_zero_floor(self):
  T=np.zeros((3,20,168),np.float32);s=f.statistics(T,np.ones(3,bool));np.testing.assert_array_equal(s["temporal_scale"],1);self.assertTrue(s["temporal_floor"].all())
 def test_U_excluded_same_window_weight(self):
  z=np.random.default_rng(1).normal(size=(3,20,368));y=np.zeros((3,20),int);y[:,0]=1;y[:,2]=-1;t=np.array(["a","a","b"]);p=k.build_pairs(z,y,t,guard=guard);b=z.copy();b[:,2]+=100;q=k.build_pairs(b,y,t,guard=guard);np.testing.assert_array_equal(p.delta,q.delta);self.assertAlmostEqual(p.weight.sum(),60)
 def test_objective_FD_fixed_epsilon(self):
  rng=np.random.default_rng(2);z=rng.normal(size=(3,20,368));y=np.zeros((3,20),int);y[:,:2]=1;y[:,4]=-1;p=k.build_pairs(z,y,np.array(["a","b","c"]),guard=guard);w=np.sin(np.arange(368)*.17)*.03;v,g,H=k.pair_objective(p,w);x,yy,sw=k.symmetric_rows(p);vs,gs,hs=k.logistic_objective(x,yy,sw,w,fit_intercept=False)
  np.testing.assert_allclose(v,vs,rtol=0,atol=1e-12);np.testing.assert_allclose(g,gs,rtol=0,atol=1e-12);np.testing.assert_allclose(H,hs,rtol=0,atol=1e-12)
  for j in range(3):
   a=np.cos(np.arange(368)*(.11+j*.07));a/=np.linalg.norm(a);eps=1e-5;plus=k.pair_objective(p,w+eps*a);minus=k.pair_objective(p,w-eps*a)
   self.assertLessEqual(abs((plus[0]-minus[0])/(2*eps)-g@a)/(1+abs(g@a)),1e-8);self.assertLessEqual(np.max(abs((plus[1]-minus[1])/(2*eps)-H@a)/(1+abs(H@a))),1e-8)
if __name__=="__main__":unittest.main(verbosity=2)
