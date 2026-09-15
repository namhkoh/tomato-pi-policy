import io,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from sim_data.native_lossless_codec import encode,decode,pack_file,load_array,MAGIC,MAX_BYTES

try:import zstandard
except ImportError:zstandard=None


@unittest.skipIf(zstandard is None,'Optional zstandard unavailable')
class NativeLosslessCodecTests(unittest.TestCase):
    def raw(self,array):
        f=io.BytesIO();np.save(f,array,allow_pickle=False);return f.getvalue()

    def test_all_numeric_widths_and_shapes_exact(self):
        rng=np.random.default_rng(84)
        for dtype in ('u1','u2','u4','u8','i1','i2','i4','i8','f2','f4','f8','>f4'):
            for shape in ((1,1),(3,7),(47,69)):
                with self.subTest(dtype=dtype,shape=shape):
                    a=rng.uniform(0,100,shape).astype(dtype);raw=self.raw(a)
                    self.assertEqual(decode(encode(raw)),raw)

    def test_preserve_float_nan_and_signed_zero_bits(self):
        bits=np.array([[0,0x80000000,0x7fc01234],[0x7f800000,0xff800000,0x3f800000]],dtype=np.uint32)
        raw=self.raw(bits.view(np.float32));decoded=decode(encode(raw))
        self.assertEqual(decoded,raw)
        self.assertEqual(np.load(io.BytesIO(decoded),allow_pickle=False).tobytes(),bits.tobytes())

    def test_create_only_and_reader(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'native.npy';b=Path(d)/'native.ghnpy';raw=self.raw(np.ones((20,20),np.float32));a.write_bytes(raw)
            r=pack_file(a,b);self.assertEqual(a.read_bytes(),raw);self.assertFalse(r['depth_recomputed'])
            np.testing.assert_array_equal(load_array(a),load_array(b))
            with self.assertRaises(FileExistsError):pack_file(a,b)
            with self.assertRaises(ValueError):pack_file(a,a)

    def test_corruption_and_truncation(self):
        packed=encode(self.raw(np.ones((5,5),np.float32)))
        for broken in [b'',packed[:10],packed[:-1],packed+b'extra',packed[:-1]+bytes([packed[-1]^1])]:
            with self.subTest(length=len(broken)):
                with self.assertRaises(ValueError):decode(broken)

    def test_uncompressed_allocation_bound(self):
        packed=encode(self.raw(np.ones((5,5),np.float32)));n=struct.unpack('<I',packed[8:12])[0]
        header=json.loads(packed[12:12+n]);header['original_size']=MAX_BYTES+1
        h=json.dumps(header).encode();broken=MAGIC+struct.pack('<I',len(h))+h+packed[12+n:]
        with self.assertRaises(ValueError):decode(broken)

    def test_non_image_and_object_arrays_rejected(self):
        for a in [np.ones(3),np.ones((0,3)),np.zeros((2,2),dtype=[('x','f4')])]:
            with self.assertRaises(ValueError):encode(self.raw(a))
        f=io.BytesIO();np.save(f,np.array([[object()]],object),allow_pickle=True)
        with self.assertRaises(ValueError):encode(f.getvalue())
        with self.assertRaises(ValueError):encode(b'not npy')

    def test_malicious_npy_shape_and_trailing_bytes_rejected(self):
        stream=io.BytesIO()
        np.lib.format.write_array_header_1_0(stream,dict(descr='<f4',fortran_order=False,shape=(1000000000,1000000000)))
        with self.assertRaises(ValueError):encode(stream.getvalue())
        with self.assertRaises(ValueError):encode(self.raw(np.ones((2,2),np.float32))+b'extra')


if __name__=='__main__':unittest.main()
