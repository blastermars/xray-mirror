# X-Ray Mirror

Point a webcam at your face and see the skull, teeth, neck vertebrae, brain and muscles inside your
head, fitted to your own face and tracked live, with your collarbones, shoulder blades and ribs below.
Open your mouth and the jaw opens. Hold up a hand and you see its bones.

Everything runs in the browser. Video never leaves the device.

## Run it

```bash
bun install
bun run dev
```

Open the printed `localhost` address in Chrome and press **Start camera**.
`bun run build` writes a static site to `dist/` that can be hosted anywhere (the camera needs HTTPS or localhost).

## On a phone

Browsers only allow the camera on HTTPS (or localhost), so a phone cannot use the plain dev address.

```bash
bun run phone
```

builds the site and serves it over HTTPS on your Wi-Fi; it prints a `Network` address such as
`https://192.168.x.x:5184/`. Open that on a phone on the same network. The certificate is
self-signed, so the phone warns once ("This connection is not private"): choose to continue, then
allow the camera. For use away from your Wi-Fi, host the `dist/` folder on any HTTPS static host.

Phones start in **Light** quality (60 % of the pixels per side, no outline model, hand and body
trackers half as often); change it under Advanced → Quality. Anywhere, Automatic drops to Light if
the frame rate stays under 17 for four seconds.

## What makes it fit

A one-size skull scaled onto a face never lines up, so the anatomy here is reshaped to the viewer:

1. **Your own face shape.** Neutral, roughly frontal frames are averaged into a 3D shape of your face.
   A thin-plate spline then warps the whole template (skull, jaw, teeth, brain, muscles, skin) so its
   skin landmarks land on yours. The warp runs in the vertex shader (`src/render/warp.ts`).
2. **Your own outline.** A person/hair segmentation of a crop around the head shows where the head
   really ends; the cranial vault is scaled until the model's outline matches it. Under hair the outline
   only acts as an upper bound (`src/fit/vault.ts`).
3. **A pose from 2D, not from guessed depth.** The head pose is solved by fitting your learned 3D shape
   to the 2D landmarks (`refinePose` in `src/tracking/faceGeometry.ts`), using only points that do not
   move with expression, so talking does not tilt the skull.
4. **A measured jaw.** The opening angle is found from your chin landmarks against your own neutral
   shape, as a hinge at the condyles plus forward glide (`src/fit/jaw.ts`).
5. **Shoulders from the body tracker.** Collarbones, shoulder blades, ribs and breastbone move as one
   block: turned by the tracked shoulder line, sized by where the shoulders appear, and set at the depth
   of the base of the neck. The neck vertebrae are bent to meet it, each upper arm bone ends on its
   tracked elbow, and the forearm runs from there to the wrist, turning with the hand when the hand
   is tracked (`src/fit/torso.ts`).
6. **Same frame for picture and overlay.** The overlay is drawn over the exact video frame that was
   analysed, so it cannot lag behind the image.

The **Fit to you** panel shows the average miss of the fitted model on tracked points next to the
miss a one-size template would have.

## The radiograph look

Anatomy is not drawn as colour. Each surface adds optical depth to a float buffer (additive, no depth
test): a cortical-shell term that grows as the surface turns edge-on, and a thickness term through
closed meshes. The film pass turns that into brightness with `1 − exp(−τ)`, then adds unsharp
masking, grain and a cool film tint (`src/render/materials.ts`, `src/render/pipeline.ts`).
Brain, muscles and eyes are not visible on a real plain film; they are optional tinted layers.

## Layout

```
src/tracking   camera, trackers, pose solve, filters
src/fit        personal shape + spline, vault outline fit, jaw, neck, chest, hands
src/render     warp, materials, frame pipeline, anatomy loading, hand and chest rigs
src/ui         controls and readouts
src/dev        synthetic test subject and scoring
tools          asset build (BodyParts3D → GLB) and template registration
tests          unit tests for the maths
```

## Rebuilding the assets

```bash
node tools/build-assets.mjs        # downloads BodyParts3D 4.0 and 3.0 into assets-src/ and writes public/assets/*.glb
bun run dev                        # then open /tools/register.html once: it writes public/assets/template.json
```

The registration page renders the template head, runs the same face tracker on it, and defines the
template's head frame the way the tracker perceives it.

## Testing

```bash
bun run test        # maths: pose, spline, jaw, hand placement, outline fit, filters
```

`/?synthetic` replaces the camera with a rendered head that has been reshaped into a different
"person" and moves around. Because its shape and motion are known, `__xray.score()` and
`__xray.regions()` in the console report how far the fitted skull is from the truth.
Add `&still` to hold it frontal, `&reshape=0` to test the template's own proportions.

## Limits

- The top and back of the skull are hidden by hair and are estimated from the outline and from
  proportions, not measured.
- Depth comes from one camera. Parts far behind the face (back of the skull) drift by a few
  millimetres when the head is turned far.
- Nerves: BodyParts3D only models the optic pathway and the nerves of the eye sockets. There is no
  spinal cord, brachial plexus or arm nerve in the data, so the Nerves layer shows those head nerves
  only (the rig in `src/render/nerves.ts` already handles cord, neck, chest and arm nerves if meshes
  named `nerve_cord`, `nerve_neck`, `nerve_chest`, `nerve_arm_l/r` are added).
- The chest is one rigid block: shrugging or rolling one shoulder is not followed.
- Arms need the elbow in view. With the elbow out of frame the upper arm hangs at rest and the forearm
  is only drawn near the hand.
- The camera's field of view is assumed from the frame shape. If the skull looks too large or small
  at the edges when you turn, adjust **Camera angle** under Advanced.
- This is an illustration built from one reference anatomy. It is not an image of your body and not
  a medical device.

See [CREDITS.md](CREDITS.md) for data sources and licences.
