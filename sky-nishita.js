// Cycles Nishita Sky Model — faithful 1:1 port from Blender's
// intern/cycles/kernel/osl/shaders/node_sky_texture.osl +
// intern/sky/source/sky_single_scattering.cpp
//
// Produces CIE XYZ sky textures; the shader converts to RGB at runtime.

// ---------------------------------------------------------------------------
// Constants (exact C++ values)
// ---------------------------------------------------------------------------

const RAYLEIGH_SCALE = 8e3;          // m
const MIE_SCALE = 1.2e3;            // m
const MIE_COEFF = 2e-5;
const MIE_G = 0.76;
const SQR_G = MIE_G * MIE_G;        // 0.5776
export const EARTH_RADIUS = 6360e3;           // m
export const ATMOSPHERE_RADIUS = 6420e3;      // m
const STEPS = 32;
const NUM_WAVELENGTHS = 21;
const MIN_WAVELENGTH = 380;
const MAX_WAVELENGTH = 780;
const STEP_LAMBDA = (MAX_WAVELENGTH - MIN_WAVELENGTH) / (NUM_WAVELENGTHS - 1); // 20

const PI = Math.PI;
const HALF_PI = PI * 0.5;
const TWO_PI = PI * 2.0;

// ---------------------------------------------------------------------------
// Spectral data tables (copied verbatim from Cycles C++)
// ---------------------------------------------------------------------------

// Solar irradiance at each wavelength (W/m^2/nm)
const IRRADIANCE = new Float64Array([
  1.45756829, 1.56596305, 1.65148449, 1.71438265, 1.78944619,
  1.69150770, 1.44647294, 1.68590050, 1.71575839, 1.67509720,
  1.63173122, 1.59429868, 1.55022753, 1.51411278, 1.47460598,
  1.43312555, 1.39261218, 1.34534425, 1.30914183, 1.27162914,
  1.24142996,
]);

// Rayleigh scattering coefficients (m^-1) at each wavelength
const RAYLEIGH_COEFF = new Float64Array([
  5.75501028e-06, 4.68782424e-06, 3.83727249e-06, 3.15613652e-06, 2.60752540e-06,
  2.16248312e-06, 1.80053120e-06, 1.50487199e-06, 1.26238665e-06, 1.06299628e-06,
  8.97693740e-07, 7.60696226e-07, 6.46595580e-07, 5.51389803e-07, 4.71615837e-07,
  4.04851729e-07, 3.48697369e-07, 3.01322282e-07, 2.61091855e-07, 2.26921510e-07,
  1.97897552e-07,
]);

// Ozone absorption coefficients (m^-1) at each wavelength
const OZONE_COEFF = new Float64Array([
  2.50458680e-28, 5.24541368e-28, 1.05608989e-27, 2.02480525e-27, 3.32547951e-27,
  4.66505840e-27, 5.11484541e-27, 4.05996633e-27, 2.42853455e-27, 1.17916386e-27,
  4.94825193e-28, 1.84610581e-28, 6.55218678e-29, 2.56992607e-29, 1.37539054e-29,
  7.27113923e-30, 2.76470970e-30, 6.34782807e-31, 7.11286337e-32, 4.52432947e-34,
  3.46700795e-38,
]);

// CIE 1931 colour matching functions (integrated per wavelength band)
// Each row = [X, Y, Z]
const CMF_XYZ = [
  new Float64Array([0.00006469, 0.00000184, 0.00030502]),
  new Float64Array([0.00108360, 0.00003900, 0.00508730]),
  new Float64Array([0.01143610, 0.00036690, 0.05374280]),
  new Float64Array([0.06325600, 0.00310500, 0.27228820]),
  new Float64Array([0.16549500, 0.02302400, 0.75721000]),
  new Float64Array([0.21007600, 0.06469600, 0.99579500]),
  new Float64Array([0.13287200, 0.13550700, 0.81651400]),
  new Float64Array([0.04117600, 0.21560600, 0.46545600]),
  new Float64Array([0.00535200, 0.31260200, 0.17084900]),
  new Float64Array([0.03079900, 0.40724500, 0.05780900]),
  new Float64Array([0.11075000, 0.50362600, 0.01581700]),
  new Float64Array([0.22554200, 0.58346100, 0.00378500]),
  new Float64Array([0.35620400, 0.63197500, 0.00087400]),
  new Float64Array([0.48427800, 0.63890300, 0.00021500]),
  new Float64Array([0.59754400, 0.60589000, 0.00004300]),
  new Float64Array([0.68653100, 0.53610500, 0.00000000]),
  new Float64Array([0.73431600, 0.44459200, 0.00000000]),
  new Float64Array([0.72285800, 0.34456800, 0.00000000]),
  new Float64Array([0.63723300, 0.25221500, 0.00000000]),
  new Float64Array([0.50453600, 0.18452200, 0.00000000]),
  new Float64Array([0.36518600, 0.12458900, 0.00000000]),
];

// 6-point Gauss-Laguerre quadrature (nodes & weights)
// Exact values from Cycles sky_single_scattering.cpp (8-point Gauss-Laguerre)
const QUADRATURE_STEPS = 8;
const QUADRATURE_NODES = new Float64Array([
  0.006811185292, 0.03614807107, 0.09004346519, 0.1706680068,
  0.2818362161, 0.4303406404, 0.6296271457, 0.9145252695,
]);
const QUADRATURE_WEIGHTS = new Float64Array([
  0.01750893642, 0.04135477391, 0.06678839063, 0.09507698807,
  0.1283416365, 0.1707430204, 0.2327233347, 0.3562490486,
]);

// ---------------------------------------------------------------------------
// Density functions
// ---------------------------------------------------------------------------

function density_rayleigh(h) {
  return Math.exp(-h / RAYLEIGH_SCALE);
}

function density_mie(h) {
  return Math.exp(-h / MIE_SCALE);
}

function density_ozone(h) {
  // Triangular profile: peaks at 25 km, spans 15–35 km
  if (h < 10000.0 || h >= 40000.0) return 0.0;
  if (h < 25000.0) return (h - 10000.0) / 15000.0;
  return 1.0 - (h - 25000.0) / 15000.0;
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

function phase_rayleigh(mu) {
  return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

function phase_mie(mu) {
  // Henyey-Greenstein
  return (3.0 / (8.0 * PI)) * ((1.0 - SQR_G) * (1.0 + mu * mu))
       / ((2.0 + SQR_G) * Math.pow(1.0 + SQR_G - 2.0 * MIE_G * mu, 1.5));
}

// ---------------------------------------------------------------------------
// geographical_to_direction  (latitude, longitude → unit direction)
// ---------------------------------------------------------------------------

function geographical_to_direction(lat, lon) {
  return [
    Math.cos(lat) * Math.cos(lon),
    Math.cos(lat) * Math.sin(lon),
    Math.sin(lat),
  ];
}

// ---------------------------------------------------------------------------
// ray_optical_depth — Gauss-Laguerre quadrature along a ray
// ---------------------------------------------------------------------------

function ray_optical_depth(ray_origin, ray_dir) {
  // Intersect ray with atmosphere sphere
  const a = ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2];
  const b = 2.0 * (ray_origin[0] * ray_dir[0] + ray_origin[1] * ray_dir[1] + ray_origin[2] * ray_dir[2]);
  const c = (ray_origin[0] * ray_origin[0] + ray_origin[1] * ray_origin[1] + ray_origin[2] * ray_origin[2])
          - ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS;
  let discr = b * b - 4.0 * a * c;

  if (discr < 0.0) {
    return new Float64Array(NUM_WAVELENGTHS);
  }

  discr = Math.sqrt(discr);
  const t_max = (-b + discr) / (2.0 * a);
  if (t_max < 0.0) {
    return new Float64Array(NUM_WAVELENGTHS);
  }

  // Returns [rayleigh, mie, ozone] density integrals (like Cycles float3)
  const ray_length = t_max;
  // optical_depth = [rayleigh_density, mie_density, ozone_density]
  let od_r = 0.0, od_m = 0.0, od_o = 0.0;

  for (let i = 0; i < QUADRATURE_STEPS; i++) {
    const px = ray_origin[0] + QUADRATURE_NODES[i] * ray_length * ray_dir[0];
    const py = ray_origin[1] + QUADRATURE_NODES[i] * ray_length * ray_dir[1];
    const pz = ray_origin[2] + QUADRATURE_NODES[i] * ray_length * ray_dir[2];

    const height = Math.sqrt(px * px + py * py + pz * pz) - EARTH_RADIUS;

    od_r += density_rayleigh(height) * QUADRATURE_WEIGHTS[i];
    od_m += density_mie(height) * QUADRATURE_WEIGHTS[i];
    od_o += density_ozone(height) * QUADRATURE_WEIGHTS[i];
  }

  return [od_r * ray_length, od_m * ray_length, od_o * ray_length];
}

// ---------------------------------------------------------------------------
// single_scattering — main inscattered radiance along a view ray
//   ray_dir, sun_dir : Float64Array(3)  — unit directions
//   ray_origin       : Float64Array(3)  — origin in metres (typically [0, 0, R_earth + alt])
//   air, aerosol, ozone : density multipliers (typically 1.0)
//   spectrum (out)   : Float64Array(NUM_WAVELENGTHS) — spectral radiance
// ---------------------------------------------------------------------------

// Faithful port of Cycles single_scattering() from sky_single_scattering.cpp
function single_scattering(ray_dir, sun_dir, ray_origin, air_density, aerosol_density, ozone_density, spectrum) {
  for (let i = 0; i < NUM_WAVELENGTHS; i++) spectrum[i] = 0.0;

  // Atmosphere intersection
  const ray_end = atmosphere_intersection(ray_origin, ray_dir);
  if (!ray_end) return;
  const ray_length = vec3_dist(ray_origin, ray_end);
  const segment_length = ray_length / STEPS;

  // Optical depth along view ray (3 components: rayleigh, mie, ozone)
  let od_r = 0.0, od_m = 0.0, od_o = 0.0;

  const mu = vec3_dot(ray_dir, sun_dir);
  const pr = phase_rayleigh(mu);
  const pm = phase_mie(mu);

  // Step along ray from origin
  for (let i = 0; i < STEPS; i++) {
    const t = (i + 0.5) * segment_length;
    const px = ray_origin[0] + ray_dir[0] * t;
    const py = ray_origin[1] + ray_dir[1] * t;
    const pz = ray_origin[2] + ray_dir[2] * t;
    const height = Math.sqrt(px * px + py * py + pz * pz) - EARTH_RADIUS;

    // Density at this point (scaled by user params)
    const dr = density_rayleigh(height) * air_density;
    const dm = density_mie(height) * aerosol_density;
    const doz = density_ozone(height) * ozone_density;

    // Accumulate optical depth along view ray
    od_r += dr * segment_length;
    od_m += dm * segment_length;
    od_o += doz * segment_length;

    // Check if Earth blocks the Sun from this point
    if (!surface_intersection([px, py, pz], sun_dir)) {
      // Optical depth from this point toward the Sun
      const light_od = ray_optical_depth([px, py, pz], sun_dir);
      const lod_r = light_od[0] * air_density;
      const lod_m = light_od[1] * aerosol_density;
      const lod_o = light_od[2] * ozone_density;

      // Total optical depth = view path + sun path
      const tot_r = od_r + lod_r;
      const tot_m = od_m + lod_m;
      const tot_o = od_o + lod_o;

      for (let wl = 0; wl < NUM_WAVELENGTHS; wl++) {
        const extinction = RAYLEIGH_COEFF[wl] * tot_r + 1.11 * MIE_COEFF * tot_m + OZONE_COEFF[wl] * tot_o;
        const attenuation = Math.exp(-extinction);
        const scattering = RAYLEIGH_COEFF[wl] * dr * pr + MIE_COEFF * dm * pm;
        spectrum[wl] += attenuation * scattering * IRRADIANCE[wl] * segment_length;
      }
    }
  }
}

// Vector helpers
function vec3_dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function vec3_dist(a, b) { const dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2]; return Math.sqrt(dx*dx+dy*dy+dz*dz); }
function vec3_len(a) { return Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); }

function surface_intersection(pos, dir) {
  if (dir[2] >= 0) return false;
  const b = 2.0 * (pos[0]*dir[0] + pos[1]*dir[1] + pos[2]*dir[2]);
  const c = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] - EARTH_RADIUS * EARTH_RADIUS;
  return (b * b - 4.0 * c) >= 0.0;
}

function atmosphere_intersection(pos, dir) {
  const b = 2.0 * (pos[0]*dir[0] + pos[1]*dir[1] + pos[2]*dir[2]);
  const c = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] - ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS;
  const d = b * b - 4.0 * c;
  if (d < 0) return null;
  const t = (-b + Math.sqrt(d)) / 2.0;
  return [pos[0]+dir[0]*t, pos[1]+dir[1]*t, pos[2]+dir[2]*t];
}

// ---------------------------------------------------------------------------
// spec_to_xyz — spectral power distribution → CIE XYZ
// ---------------------------------------------------------------------------

function spec_to_xyz(spectrum) {
  let x = 0.0, y = 0.0, z = 0.0;
  for (let i = 0; i < NUM_WAVELENGTHS; i++) {
    x += CMF_XYZ[i][0] * spectrum[i] * STEP_LAMBDA;
    y += CMF_XYZ[i][1] * spectrum[i] * STEP_LAMBDA;
    z += CMF_XYZ[i][2] * spectrum[i] * STEP_LAMBDA;
  }
  return [x, y, z];
}

// ---------------------------------------------------------------------------
// earthIntersectionAngle — angle below horizon where the Earth surface
// becomes visible from a given altitude (metres above ground)
// ---------------------------------------------------------------------------

export function earthIntersectionAngle(altitude) {
  if (altitude <= 0.0) return 0.0;
  const R = EARTH_RADIUS;
  const h = altitude;
  return Math.acos(R / (R + h));
}

// ---------------------------------------------------------------------------
// precomputeSkyTexture
//
// Fills a Float32Array of (width * height * 4) with XYZ + 0 values.
// Stride is 4 for direct upload to WebGPU rgba32float / rgba16float.
//
// UV mapping matches Cycles exactly:
//   longitude = 2π * (x / width)                     [0, 2π)
//   For upper half (y >= half_height):
//     latitude = π/2 * sqr((y - half_height) / half_height)    [0, π/2]
//   For lower half (y < half_height):
//     copies the horizon row and fades toward 0
//
// The sun is placed in the XZ plane at the given elevation.
// ---------------------------------------------------------------------------

export function precomputeSkyTexture(width, height, sunElevation, altitude, airDensity, aerosolDensity, ozoneDensity) {
  const stride = 4;
  const pixels = new Float32Array(width * height * stride);

  const half_height = height / 2;

  // Ray origin: observer at Earth centre + radius + altitude
  const ray_origin = [0.0, 0.0, EARTH_RADIUS + altitude];

  // Sun direction from elevation (in XZ plane)
  const sun_dir = [
    Math.cos(sunElevation),
    0.0,
    Math.sin(sunElevation),
  ];

  const spectrum = new Float64Array(NUM_WAVELENGTHS);

  // --- Upper hemisphere (y >= half_height) ---
  for (let y = half_height; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Latitude: non-linear mapping  lat = π/2 * sqr(t) where t = (y - half_height) / half_height
      const t = (y - half_height) / half_height;
      const latitude = HALF_PI * t * t;
      // Longitude
      const longitude = TWO_PI * x / width;

      const ray_dir = geographical_to_direction(latitude, longitude);

      single_scattering(ray_dir, sun_dir, ray_origin, airDensity, aerosolDensity, ozoneDensity, spectrum);
      const xyz = spec_to_xyz(spectrum);

      const idx = (y * width + x) * stride;
      pixels[idx + 0] = xyz[0];
      pixels[idx + 1] = xyz[1];
      pixels[idx + 2] = xyz[2];
      pixels[idx + 3] = 0.0;
    }
  }

  // --- Lower hemisphere (y < half_height): fade from horizon row ---
  // The horizon row is at y = half_height (latitude = 0)
  // For rows below, we interpolate toward black.
  const horizon_y = half_height; // first row of upper hemisphere = horizon
  for (let y = 0; y < half_height; y++) {
    const fade = y / half_height; // 0 at bottom, 1 at horizon
    for (let x = 0; x < width; x++) {
      const src = (horizon_y * width + x) * stride;
      const dst = (y * width + x) * stride;
      pixels[dst + 0] = pixels[src + 0] * fade;
      pixels[dst + 1] = pixels[src + 1] * fade;
      pixels[dst + 2] = pixels[src + 2] * fade;
      pixels[dst + 3] = 0.0;
    }
  }

  return pixels;
}

// ---------------------------------------------------------------------------
// precomputeSunDisc
//
// Computes atmospheric transmittance for two directions:
//   - bottom of sun disc (elevation - angularDiameter/2)
//   - top of sun disc    (elevation + angularDiameter/2)
// Returns XYZ values for each, so the shader can interpolate.
// ---------------------------------------------------------------------------

// Faithful port of Cycles SKY_single_scattering_precompute_sun
export function precomputeSunDisc(sunElevation, angularDiameter, altitude, airDensity, aerosolDensity) {
  altitude = Math.max(1.0, Math.min(altitude, 59999.0));
  const half_angular = angularDiameter / 2.0;
  const solid_angle = 2.0 * Math.PI * (1.0 - Math.cos(half_angular));
  const cam_pos = [0.0, 0.0, EARTH_RADIUS + altitude];

  function sun_radiation(cam_dir) {
    const od = ray_optical_depth(cam_pos, cam_dir);
    const spectrum = new Float64Array(NUM_WAVELENGTHS);
    for (let i = 0; i < NUM_WAVELENGTHS; i++) {
      const transmittance = RAYLEIGH_COEFF[i] * od[0] * airDensity +
                            1.11 * MIE_COEFF * od[1] * aerosolDensity;
      spectrum[i] = IRRADIANCE[i] * Math.exp(-transmittance) / solid_angle;
    }
    return spec_to_xyz(spectrum);
  }

  const bottom = sunElevation - half_angular;
  const top = sunElevation + half_angular;

  if (top <= 0.0) return { bottomXYZ: [0,0,0], topXYZ: [0,0,0] };

  const elBottom = Math.max(bottom, 0.0);
  const elTop = Math.max(top, 0.0);

  const bottomXYZ = sun_radiation(geographical_to_direction(elBottom, 0.0));
  const topXYZ = sun_radiation(geographical_to_direction(elTop, 0.0));

  return { bottomXYZ, topXYZ };
}
