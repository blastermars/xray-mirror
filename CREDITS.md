# Credits

## Anatomy

**BodyParts3D 4.0**, © The Database Center for Life Science, licensed under
[CC Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Source: https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/

Used for the skull, mandible, teeth, cervical vertebrae, hyoid, brain, eyes, neck and eye muscles,
skin envelope, hand bones, the collarbones, shoulder blades, ribs, breastbone, thoracic vertebrae
and upper arm bones, and the nerves of the eye sockets and visual pathway (`public/assets/head.glb`,
`public/assets/hand.glb`, `public/assets/torso.glb`, `public/assets/nerves.glb`).

**BodyParts3D 3.0**, Copyright© The Database Center for Life Science, licensed by
[CC Attribution-Share Alike 2.1 Japan](http://creativecommons.org/licenses/by-sa/2.1/jp/deed.en_US).
Source: https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/

Used for the muscles of the face and of chewing (`public/assets/muscles.glb`), which release 4.0 does
not contain. That file is a derivative of the 3.0 data and is distributed under the same
CC BY-SA 2.1 Japan licence. (DBCLS's current licence page names CC BY 4.0 for the database but does
not say whether that covers earlier releases, so the stricter share-alike terms are kept for this file.)
Changes made to it: muscles selected, simplified, left and right merged, and scaled and moved
(uniform scale 1.059 plus a translation, fitted on shared skull bones) into the 4.0 coordinate frame.

Changes made to the 4.0 data: meshes were selected, simplified, merged per structure and compressed
(`tools/build-assets.mjs`); the eyeballs are closed envelopes rebuilt from the original eye parts.
At run time the meshes are moved into the tracker's head frame and reshaped to the viewer.

## Tracking

[MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/guide) (Apache 2.0): face
landmarker, hand landmarker and selfie multiclass segmenter, loaded at run time.
`src/data/canonicalFace.json` is MediaPipe's canonical face model (Apache 2.0).
`src/tracking/faceGeometry.ts` is a port of MediaPipe's face-geometry pipeline with changes.

## Rendering

[three.js](https://threejs.org/) (MIT).

## References used for the design

- Hona & Stephan, "Global facial soft tissue thicknesses for craniofacial identification (2023)",
  Int J Legal Med 2024: soft-tissue depths used to sanity-check the template registration.
- Casiez, Roussel & Vogel, "1€ Filter", CHI 2012: pose smoothing.
