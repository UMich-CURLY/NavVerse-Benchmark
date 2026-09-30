# NavVerse Real-World Experiment Server Sites relay

This is the Sites deployment surface for `navverse_tools.real_world_exp_server`.

- The mobile UI is synchronized from `../static` by `npm run phone:sync`.
- D1 stores episode and action metadata.
- The bridge registers selectable local agent profiles and each episode keeps its selection.
- R2 temporarily holds each JPEG until the outbound local bridge downloads it.
- `python -m navverse_tools.real_world_exp_server.sites_bridge` runs the navigation model locally and returns actions.

Build and validate with Node 22 or newer:

```bash
npm ci
npm test
npm run lint
```

The checked-in hosting metadata contains binding names only. Configure your own
project identifier and cloud resources before deployment; this source does
not point to the original private deployment. Database migrations are already
included; regenerate them only when changing the schema.
