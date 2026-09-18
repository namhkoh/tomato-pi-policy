"""Execute the installer's actual partition guard; no alternate predicate."""
import ast
import inspect
import unittest
from copy import deepcopy
from . import native848_persistent_far_population_v1 as p
from . import native848_persistent_far_overlay_v1 as reuse


def partition(recipe, current_roots):
    body=ast.parse(inspect.getsource(p.install_overlay)).body[0].body
    first=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='whole' for t in n.targets))
    last=next(i for i,n in enumerate(body[first:],first) if isinstance(n,ast.Expr)
              and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant)
              and a.value=='Runtime population root set differs' for a in n.value.args))
    program=ast.fix_missing_locations(ast.Module(body=body[first:last+1],type_ignores=[]))
    scope=dict(recipe=recipe,current_roots=current_roots,primary='P0',require=p.require)
    exec(compile(program,'<actual installer partition guards>','exec'),scope)
    return scope


def recipe(whole_count):
    rows=[dict(plant_root='P'+str(i),mode='original_foreground' if i==0 else
          'complete_original_components' if i<=whole_count else 'unchanged_packed_background') for i in range(144)]
    return dict(plants=rows,whole_original_background_roots=['P'+str(i) for i in range(1,whole_count+1)],
                whole_background_count=whole_count,unchanged_packed_background_count=143-whole_count)


class PartitionTests(unittest.TestCase):
    def test_full144_partition_boundaries_and_dense9_case(self):
        for count in (0,6,9,143):
            with self.subTest(count=count):
                r=recipe(count);s=partition(r,{'P'+str(i) for i in range(144)})
                self.assertEqual(len(s['packed']),143-count)

    def test_declared_counts_cannot_hide_missing_or_extra_shapes(self):
        for key in ('whole_background_count','unchanged_packed_background_count'):
            r=recipe(9);r[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):partition(r,set('P'+str(i) for i in range(144)))

    def test_foreground_cannot_be_packed(self):
        r=recipe(9);r['plants'][0]['mode']='unchanged_packed_background'
        with self.assertRaises(ValueError):partition(r,set('P'+str(i) for i in range(144)))

    def test_whole_plant_cannot_overlap_packed_set(self):
        r=recipe(9);r['whole_original_background_roots'][-1]='P10'
        with self.assertRaises(ValueError):partition(r,set('P'+str(i) for i in range(144)))

    def test_missing_slot_and_runtime_root_substitution_rejected(self):
        r=recipe(9);r['plants'].pop()
        with self.assertRaises(ValueError):partition(r,set('P'+str(i) for i in range(144)))
        roots=set('P'+str(i) for i in range(143))|{'Other'}
        with self.assertRaises(ValueError):partition(recipe(9),roots)

    def test_dispatch_requires_exact_recipe_type(self):
        self.assertIs(reuse.recipe_installer(p.RECIPE_SCHEMA),p.install_overlay)
        self.assertIs(reuse.recipe_installer(reuse.frozen.finite.SCHEMA),reuse.frozen.install_overlay)
        with self.assertRaises(ValueError):reuse.recipe_installer('untyped-or-renamed')


if __name__=='__main__':unittest.main(verbosity=2)
