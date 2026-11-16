// Output mesh definitions: which BodyParts3D 4.0 ELEMENT files (selected by the exact English name in the
// OBJ header, case-insensitive) go into each named output mesh, plus triangle budget and simplification error cap.
// errFrac: max simplification error as a fraction of each source element's largest bbox side;
// errMax: absolute cap in mm. Budgets are split across a group's elements in proportion to surface area.

const LR = (n) => [`Left ${n}`, `Right ${n}`];

const CRANIUM = [
  'Frontal bone', ...LR('parietal bone'), 'Occipital bone', ...LR('temporal bone'), 'Sphenoid bone', 'Ethmoid',
  ...LR('maxilla'), ...LR('zygomatic bone'), ...LR('nasal bone'), ...LR('lacrimal bone'), ...LR('palatine bone'),
  'Vomer', ...LR('inferior nasal concha'),
];

const TOOTH_TYPES = ['central secondary incisor tooth', 'lateral secondary incisor tooth', 'secondary canine tooth',
  'first secondary premolar tooth', 'second secondary premolar tooth', 'first secondary molar tooth', 'second secondary molar tooth'];
const teeth = (jaw) => TOOTH_TYPES.flatMap((t) => [`Left ${jaw} ${t}`, `Right ${jaw} ${t}`]);

const CEREBRAL_CORTEX_GYRI = [
  'superior frontal gyrus', 'middle frontal gyrus', 'inferior frontal gyrus', 'precentral gyrus', 'postcentral gyrus',
  'superior parietal lobule', 'supramarginal gyrus', 'angular gyrus', 'occipital lobe', 'anterior part of {} superior temporal gyrus',
  'posterior part of {} superior temporal gyrus', 'middle temporal gyrus', 'inferior temporal gyrus', 'fusiform gyrus',
  'parahippocampal gyrus', 'cingulate gyrus', 'insula',
];
const gyri = CEREBRAL_CORTEX_GYRI.flatMap((g) => g.includes('{}')
  ? [g.replace('{}', 'left'), g.replace('{}', 'right')].map((s) => s[0].toUpperCase() + s.slice(1))
  : LR(g));

// Muscles. BodyParts3D 4.0 has NO muscles of facial expression and NO muscles of mastication
// (its "muscle of face" concept contains only the extra-ocular muscles). See report / manifest notes.
export const MUSCLES = [
  ['platysma', 'Platysma', LR('platysma')],
  ['sternocleidomastoid', 'Sternocleidomastoid', LR('sternocleidomastoid')],
  ['digastric', 'Digastric (incl. intermediate tendon)', [...LR('digastric'), 'Right intermediate tendon']],
  ['stylohyoid', 'Stylohyoid', LR('stylohyoid')],
  ['mylohyoid', 'Mylohyoid', LR('mylohyoid')],
  ['geniohyoid', 'Geniohyoid', LR('geniohyoid')],
  ['omohyoid', 'Omohyoid', LR('omohyoid')],
  ['sternohyoid', 'Sternohyoid', LR('sternohyoid')],
  ['sternothyroid', 'Sternothyroid', LR('sternothyroid')],
  ['thyrohyoid', 'Thyrohyoid', LR('thyrohyoid')],
  ['scalenus_anterior', 'Scalenus anterior', LR('scalenus anterior')],
  ['scalenus_medius', 'Scalenus medius', LR('scalenus medius')],
  ['scalenus_posterior', 'Scalenus posterior', LR('scalenus posterior')],
  ['longus_capitis', 'Longus capitis', LR('longus capitis')],
  ['longus_colli', 'Longus colli (left side only in BodyParts3D)', ['Superior oblique part of left longus colli',
    'Vertical intermediate part of left longus colli', 'Inferior oblique part of left longus colli']],
  ['levator_scapulae', 'Levator scapulae', LR('levator scapulae')],
  // extra-ocular muscles = BodyParts3D "muscle of face" (FMA46751)
  ['superior_rectus', 'Superior rectus (extra-ocular)', LR('superior rectus')],
  ['inferior_rectus', 'Inferior rectus (extra-ocular)', LR('inferior rectus')],
  ['medial_rectus', 'Medial rectus (extra-ocular)', LR('medial rectus')],
  ['lateral_rectus', 'Lateral rectus (extra-ocular)', LR('lateral rectus')],
  ['superior_oblique', 'Superior oblique (extra-ocular)', LR('superior oblique')],
  ['inferior_oblique', 'Inferior oblique (extra-ocular)', LR('inferior oblique')],
  ['levator_palpebrae_superioris', 'Levator palpebrae superioris', LR('levator palpebrae superioris')],
];

const EYE_PARTS = (side) => [`${side} sclera`, `${side} cornea`, `${side} vitreous body`, `${side} lens`,
  `Anterior chamber of ${side.toLowerCase()} eyeball`, `${side} choroid`, `Optic part of ${side.toLowerCase()} retina`];

const R = (n) => [`Right ${n}`];
const FINGERS = ['thumb', 'index finger', 'middle finger', 'ring finger', 'little finger'];
const ORD = ['first', 'second', 'third', 'fourth', 'fifth'];

const B = { errFrac: 0.01, errMax: 0.5 }; // default error cap

export const HEAD = [
  { name: 'cranium', label: 'Cranium (skull excluding mandible, hyoid and teeth)', names: CRANIUM, budget: 70000, ...B },
  { name: 'mandible', label: 'Mandible', names: ['Mandible'], budget: 12000, ...B },
  { name: 'teeth_upper', label: 'Maxillary teeth (permanent dentition, no third molars)', names: teeth('upper'), budget: 8000, errFrac: 0.01, errMax: 0.15 },
  { name: 'teeth_lower', label: 'Mandibular teeth (permanent dentition, no third molars)', names: teeth('lower'), budget: 8000, errFrac: 0.01, errMax: 0.15 },
  { name: 'hyoid', label: 'Hyoid bone', names: ['Hyoid bone'], budget: 1000, errFrac: 0.01, errMax: 0.15 },
  { name: 'c1', label: 'Atlas (C1)', names: ['Atlas'], budget: 3000, ...B },
  { name: 'c2', label: 'Axis (C2)', names: ['Axis'], budget: 3000, ...B },
  { name: 'c3', label: 'Third cervical vertebra (C3)', names: ['Third cervical vertebra'], budget: 3000, ...B },
  { name: 'c4', label: 'Fourth cervical vertebra (C4)', names: ['Fourth cervical vertebra'], budget: 3000, ...B },
  { name: 'c5', label: 'Fifth cervical vertebra (C5)', names: ['Fifth cervical vertebra'], budget: 3000, ...B },
  { name: 'c6', label: 'Sixth cervical vertebra (C6)', names: ['Sixth cervical vertebra'], budget: 3000, ...B },
  { name: 'c7', label: 'Seventh cervical vertebra (C7)', names: ['Seventh cervical vertebra'], budget: 3000, ...B },
  { name: 'cerebrum', label: 'Cerebrum (cortical gyri/lobes + hemispheric white matter core)', names: [...gyri, 'Orbital gyrus', 'White matter of left cerebral hemisphere', 'White matter of right cerebral hemisphere'], budget: 40000, errFrac: 0.01, errMax: 0.6 },
  { name: 'cerebellum', label: 'Cerebellum', names: ['Cerebellum'], budget: 10000, ...B },
  { name: 'brainstem', label: 'Brainstem (midbrain incl. cerebral peduncles and colliculi, pons, medulla oblongata)', names: ['Midbrain', 'Peduncle of midbrain', ...LR('superior colliculus'), ...LR('inferior colliculus'), 'Pons', 'Medulla oblongata'], budget: 4000, errFrac: 0.01, errMax: 0.3 },
  { name: 'eye_l', label: "Left eyeball (subject's left)", names: EYE_PARTS('Left'), budget: 1280, method: 'envelope' },
  { name: 'eye_r', label: "Right eyeball (subject's right)", names: EYE_PARTS('Right'), budget: 1280, method: 'envelope' },
  ...MUSCLES.map(([slug, label, names]) => ({ name: `muscle_${slug}`, label, names, group: 'muscles', errFrac: 0.01, errMax: 0.4 })),
  { name: 'skin', label: 'Skin of head and neck (cropped and capped)', names: ['Skin'], budget: 16000, errFrac: 0.01, errMax: 0.4, method: 'skin' },
];
export const GROUP_BUDGETS = { muscles: 30000 };

export const HAND = [
  { name: 'radius', label: 'Right radius', names: R('radius') },
  { name: 'ulna', label: 'Right ulna', names: R('ulna') },
  { name: 'scaphoid', label: 'Right scaphoid', names: R('scaphoid') },
  { name: 'lunate', label: 'Right lunate', names: R('lunate') },
  { name: 'triquetrum', label: 'Right triquetrum', names: R('triquetral') },
  { name: 'pisiform', label: 'Right pisiform', names: R('pisiform') },
  { name: 'trapezium', label: 'Right trapezium', names: R('trapezium') },
  { name: 'trapezoid', label: 'Right trapezoid', names: R('trapezoid') },
  { name: 'capitate', label: 'Right capitate', names: R('capitate') },
  { name: 'hamate', label: 'Right hamate', names: R('hamate') },
  ...ORD.map((o, i) => ({ name: `mc${i + 1}`, label: `Right ${o} metacarpal`, names: R(`${o} metacarpal bone`) })),
  ...FINGERS.map((f, i) => ({ name: `pp${i + 1}`, label: `Proximal phalanx of right ${f}`, names: [`Proximal phalanx of right ${f}`] })),
  ...FINGERS.slice(1).map((f, i) => ({ name: `mp${i + 2}`, label: `Middle phalanx of right ${f}`, names: [`Middle phalanx of right ${f}`] })),
  ...FINGERS.map((f, i) => ({ name: `dp${i + 1}`, label: `Distal phalanx of right ${f}`, names: [`Distal phalanx of right ${f}`] })),
].map((p) => ({ budget: /radius|ulna/.test(p.name) ? 3000 : /^mc/.test(p.name) ? 1500 : /^[pmd]p/.test(p.name) ? 1000 : 800, errFrac: 0.01, errMax: 0.15, ...p }));

// ---- muscles.glb: muscles of facial expression and mastication from BodyParts3D 3.0 (absent from 4.0).
// Names are 3.0 parts_list_e.txt concept names (primitive meshes). Neck / extra-ocular muscles and the
// platysma stay in head.glb (4.0) and are deliberately not duplicated here.
export const FACE_MUSCLES = [
  ['masseter', 'Masseter (superficial + deep parts)', [...LR('superficial part of {} masseter'), ...LR('deep part of {} masseter')]],
  ['temporalis', 'Temporalis', LR('{} temporalis')],
  ['medial_pterygoid', 'Medial pterygoid', LR('{} medial pterygoid')],
  ['lateral_pterygoid', 'Lateral pterygoid (upper + lower heads)', [...LR('upper head of {} lateral pterygoid'), ...LR('lower head of {} lateral pterygoid')]],
  ['buccinator', 'Buccinator', LR('{} buccinator')],
  ['orbicularis_oculi', 'Orbicularis oculi (orbital + palpebral parts)', [...LR('orbital part of {} orbicularis oculi'), ...LR('palpebral part of {} orbicularis oculi')]],
  ['orbicularis_oris', 'Orbicularis oris', ['orbicularis oris']],
  ['frontalis', 'Frontalis (frontal belly of occipitofrontalis)', LR('{} frontalis')],
  ['occipitalis', 'Occipitalis (occipital belly of occipitofrontalis)', LR('{} occipitalis')],
  ['temporoparietalis', 'Temporoparietalis', LR('{} temporoparietalis')],
  ['procerus', 'Procerus', LR('{} procerus')],
  ['nasalis', 'Nasalis', LR('{} nasalis')],
  ['depressor_septi_nasi', 'Depressor septi nasi', LR('{} depressor septi nasi')],
  ['corrugator_supercilii', 'Corrugator supercilii', LR('{} corrugator supercilii')],
  ['zygomaticus_major', 'Zygomaticus major', LR('{} zygomaticus major')],
  ['zygomaticus_minor', 'Zygomaticus minor', LR('{} zygomaticus minor')],
  ['levator_labii_superioris', 'Levator labii superioris', LR('{} levator labii superioris')],
  ['levator_labii_superioris_alaeque_nasi', 'Levator labii superioris alaeque nasi', LR('{} levator labii superioris alaeque nasi')],
  ['levator_anguli_oris', 'Levator anguli oris', LR('{} levator anguli oris')],
  ['depressor_anguli_oris', 'Depressor anguli oris', LR('{} depressor anguli oris')],
  ['depressor_labii_inferioris', 'Depressor labii inferioris', LR('{} depressor labii inferioris')],
  ['risorius', 'Risorius', LR('{} risorius')],
  ['mentalis', 'Mentalis', LR('{} mentalis')],
].map(([slug, label, names]) => ({
  name: `muscle_${slug}`, label,
  // LR() yields "Left x"/"Right x"; 3.0 names put the side inside the phrase ("deep part of left masseter")
  names: names.map((n) => (n.includes('{}') ? n.replace(/^(Left|Right) (.*)$/, (_, s, rest) => rest.replace('{}', s.toLowerCase())) : n)),
  group: 'face_muscles', errFrac: 0.01, errMax: 0.3,
}));
export const FACE_MUSCLE_BUDGET = 30000;

// ---- torso.glb (BodyParts3D 4.0, same frame as head.glb): shoulder girdle + upper chest. _l = subject's left (+x).
const RIB_ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'];
export const TORSO = [
  { name: 'clavicle_l', label: 'Left clavicle', names: ['Left clavicle'], budget: 2000 },
  { name: 'clavicle_r', label: 'Right clavicle', names: ['Right clavicle'], budget: 2000 },
  { name: 'scapula_l', label: 'Left scapula', names: ['Left scapula'], budget: 5000 },
  { name: 'scapula_r', label: 'Right scapula', names: ['Right scapula'], budget: 5000 },
  { name: 'humerus_l', label: 'Left humerus', names: ['Left humerus'], budget: 3000 },
  { name: 'humerus_r', label: 'Right humerus', names: ['Right humerus'], budget: 3000 },
  { name: 'sternum', label: 'Sternum (manubrium + body + xiphoid process)', names: ['Manubrium', 'Body of sternum', 'Xiphoid process'], budget: 4000 },
  { name: 'ribs_l', label: 'Left ribs 1-7 (bone)', names: RIB_ORD.map((o) => `Left ${o} rib`), budget: 9000 },
  { name: 'ribs_r', label: 'Right ribs 1-7 (bone)', names: RIB_ORD.map((o) => `Right ${o} rib`), budget: 9000 },
  { name: 'costal_cartilage', label: 'Costal cartilages of ribs 1-7, both sides', names: RIB_ORD.flatMap((o) => [`Left ${o} costal cartilage`, `Right ${o} costal cartilage`]), budget: 4000 },
  { name: 'skin_torso', label: 'Upper-trunk skin, outer surface only (open cut edges; soft envelope)', names: ['Skin'], budget: 6000, method: 'skinTorso' },
  { name: 'thoracic_spine', label: 'Thoracic vertebrae T1-T7 (no discs)', names: RIB_ORD.map((o) => `${o[0].toUpperCase()}${o.slice(1)} thoracic vertebra`), budget: 14000 },
].map((p) => ({ errFrac: 0.01, errMax: 0.5, ...p }));

// ---- nerves.glb (BodyParts3D 4.0). Neither 4.0 nor 3.0 models the spinal cord (only its central canal), spinal
// nerves, plexuses or any nerve of the neck, chest or arm; the only nerves present are the orbital/optic ones below.
const LR2 = (n) => [`Left ${n}`, `Right ${n}`];
export const NERVES = [
  {
    name: 'nerve_head', label: 'Cranial nerves of the orbit and visual pathway (as modelled in BodyParts3D 4.0)',
    names: [
      ...LR2('optic nerve'), 'Optic chiasm', ...LR2('optic tract'),
      ...['Superior branch of {} oculomotor nerve', 'Inferior branch of {} oculomotor nerve'].flatMap((n) => ['left', 'right'].map((s) => n.replace('{}', s))),
      ...LR2('trochlear nerve'), ...LR2('ophthalmic nerve'), ...LR2('frontal nerve'), ...LR2('supra-orbital nerve'), ...LR2('supratrochlear nerve'),
      ...LR2('lacrimal nerve'), ...LR2('nasociliary nerve'), ...LR2('infratrochlear nerve'), ...LR2('anterior ethmoidal nerve'),
      ...LR2('posterior ethmoidal nerve'), ...LR2('long ciliary nerve'), 'Short ciliary nerve',
      'Communicating branch of left nasociliary nerve with left ciliary ganglion', 'Communicating branch of right nasociliary nerve with right ciliary ganglion',
      ...LR2('ciliary ganglion'),
    ],
    // Each optic nerve exists twice in 4.0 (orbit set FJ1313/FJ1364 and brain set FJ1772/FJ1819), overlapping ~0.42 of
    // 0.63 cm3; keep the orbit-set version (consistent with the other orbital nerves, penetrates the eyeball less).
    exclude: { FJ1772: 'alternative model of the left optic nerve (duplicate of FJ1313)', FJ1819: 'alternative model of the right optic nerve (duplicate of FJ1364)' },
    budget: 1e9, errFrac: 0.01, errMax: 0.1, // keep source resolution: thin tubes
  },
];
