from copy import deepcopy
from .plant_contact_stream import PlantContactStream


def test_flat_schema_snapshot_is_lossless_and_detached_including_every_vector():
    stream=PlantContactStream(['/Plant/Stem'])
    normal=dict(collider0='/Plant/Stem',collider1='/Robot/Finger',kind='normal',
                point_world_m=[1.,2.,3.],impulse_on_0_ns=[-1e-38,.2,-.3],
                normal_on_0=[0.,0.,1.],separation_m=-.0002)
    friction={key:value for key,value in normal.items() if key not in ('normal_on_0','separation_m')}
    friction['kind']='friction'
    stream.add_contact(normal);stream.add_contact(friction)
    expected=deepcopy(stream._rows)
    rows=stream.rows
    assert rows==expected
    for row in rows:
        for value in row.values():
            if isinstance(value,list):value[0]=99
        row['collider0']='/Wrong'
    assert stream.rows==expected
    snapshot=stream.snapshot();stream.begin_step()
    assert snapshot['rows']==expected and stream.rows==[]
