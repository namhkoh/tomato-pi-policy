# Supplied greenhouse package preview

Branch: `koh-dev/sim-data`.

Source archive: `C:\Users\USER\Downloads\tomato_greenhouse_pack.tar.gz`.
Extracted unchanged into
`data/sim_data/package_20260905/tomato_greenhouse_pack/` (gitignored).
The archive contains the building, 24 component plants, 26 joined background
plants, the segmentation taxonomy, and the environment-lighting panel. It does
not include a complete assembled robot/physics application.

From the repository root on this Windows machine:

```bat
examples\greenhouse_sim\run_sim_data.cmd
```

The launcher uses `D:\isaac-sim-6.0.1\python.bat` and opens the supplied
`house/green_house_base.usd`. Override the unpacked package root if needed:

```bat
examples\greenhouse_sim\run_sim_data.cmd --package D:\path\to\tomato_greenhouse_pack
```

The 75-gutter building is retained. Three central gutters receive 144 plants
at 0.50 m spacing, x offsets of +/-0.195 m, and base height z=0.90 m.
142 are instanceable backdrop references; two are assembled component plants.
Use the **Tomato package preview** panel to select the aisle, close-up, or
overview camera. The **Greenhouse environment** panel changes time, season,
sun intensity, and sky intensity. Preview illumination starts at sun=1500,
sky=1200 to reduce the clipping described in the package guide.

All additions and light changes live in the USD session layer. No packaged
asset is saved or modified. Unbundled external prop payloads are deactivated
in that session before the local greenhouse payloads are loaded; their exact
prim paths are recorded in the status report.

The actual manifests contain plant-frame component positions. The launcher
subtracts each parent's position before authoring its child's local transform.
Treating those positions directly as parent-relative would accumulate offsets
and incorrectly stretch the assembled plant.

Runtime files: `data/sim_data/preview_20260905/` contains `status.json` and
the verified viewport capture `preview.png`. The successful initial launch
is recorded in `stdout_v3.log` / `stderr_v3.log`; the earlier logs retain
startup diagnostics.

Verified on 2026-09-05: the GUI reached `PACKAGE_READY`, remained responsive,
and rendered the planted aisle. The two component plants contain 833 parts;
their assembled bounds span roughly z=0.70-3.97 m. Fifty-one external prop
roots were excluded from the session. Material loading is asynchronous to
keep the GUI responsive.

This launch is a geometry/lighting preview. Robot attachment, physical
grasp/cut behavior, semantic-label authoring, and dataset recording have not
yet been integrated with this package. The existing VLM changes are preserved
as uncommitted edits; creating this branch did not commit them.
