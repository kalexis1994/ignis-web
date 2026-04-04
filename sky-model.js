import * as nishita from './sky-nishita.js';

const PI = Math.PI;
const HALF_PI = PI * 0.5;
const TWO_PI = PI * 2.0;
const INV_PI = 1.0 / PI;
const INV_FOUR_PI = 1.0 / (4.0 * PI);

const GROUND_ALBEDO = [0.3, 0.3, 0.3, 0.3];
const PHASE_ISOTROPIC = INV_FOUR_PI;
const RAYLEIGH_PHASE_SCALE = (3.0 / 16.0) * INV_PI;
const AEROSOL_G = 0.8;
const AEROSOL_G2 = AEROSOL_G * AEROSOL_G;
const EARTH_RADIUS_KM = 6371.0;
const ATMOSPHERE_THICKNESS_KM = 100.0;
const ATMOSPHERE_RADIUS_KM = EARTH_RADIUS_KM + ATMOSPHERE_THICKNESS_KM;
const TRANSMITTANCE_STEPS = 64;
const IN_SCATTERING_STEPS = 64;
const TRANSMITTANCE_RES_X = 256;
const TRANSMITTANCE_RES_Y = 64;

const SUN_SPECTRAL_IRRADIANCE = [1.679, 1.828, 1.986, 1.307];
const MOLECULAR_SCATTERING_COEFFICIENT_BASE = [6.605e-3, 1.067e-2, 1.842e-2, 3.156e-2];
const OZONE_ABSORPTION_CROSS_SECTION = [3.472e-25, 3.914e-25, 1.349e-25, 11.03e-27];
const OZONE_MEAN_DOBSON = 334.5;
const AEROSOL_ABSORPTION_CROSS_SECTION = [2.8722e-24, 4.6168e-24, 7.9706e-24, 1.3578e-23];
const AEROSOL_SCATTERING_CROSS_SECTION = [1.5908e-22, 1.7711e-22, 2.0942e-22, 2.4033e-22];
const AEROSOL_BASE_DENSITY = 1.3681e20;
const AEROSOL_BACKGROUND_DENSITY = 2e6;
const AEROSOL_HEIGHT_SCALE = 0.73;
const MULTI_SCATTER_FIT = [0.217, 0.347, 0.594, 1.0];

const SPECTRAL_XYZ = [
  [53.38691773856467, 22.981337506691025, 0.0],
  [43.90484446636936, 71.34779570005339, 0.1025068679657413],
  [1.6137278251608962, 18.422960591455485, 31.742921188390806],
  [20.762668673810577, 2.361421352331437, 110.4800964325214],
];

const XYZ_TO_LINEAR_SRGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.9692660, 1.8760108, 0.0415560],
  [0.0556434, -0.2040259, 1.0572252],
];

export const CYCLES_SUN_ANGULAR_DIAMETER = 0.545 * PI / 180.0;
export const CYCLES_SUN_SOLID_ANGLE = TWO_PI * (1.0 - Math.cos(CYCLES_SUN_ANGULAR_DIAMETER * 0.5));
const CYCLES_SKY_TEXTURE_WIDTH = 512;
const CYCLES_SKY_TEXTURE_HEIGHT = 256;

export const CYCLES_SKY_DEFAULTS = Object.freeze({
  altitude: 100.0,
  airDensity: 1.0,
  aerosolDensity: 1.0,
  ozoneDensity: 1.0,
  sunAngularDiameter: CYCLES_SUN_ANGULAR_DIAMETER,
  sunIntensity: 1.0,
});

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function saturate(value) {
  return clamp(value, 0.0, 1.0);
}

function safeSqrt(value) {
  return Math.sqrt(Math.max(value, 0.0));
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function fract(value) {
  return value - Math.floor(value);
}

function envUvToWorldDir(u, v) {
  const theta = v * PI;
  const phi = (0.5 - u) * TWO_PI;
  const sinTheta = Math.sin(theta);
  return [
    Math.cos(phi) * sinTheta,
    Math.cos(theta),
    -Math.sin(phi) * sinTheta,
  ];
}

function worldToCyclesLocal(dir) {
  return [dir[0], dir[2], dir[1]];
}

function sign(value) {
  return value < 0.0 ? -1.0 : (value > 0.0 ? 1.0 : 0.0);
}

function wrapAnglePositive(angle) {
  let wrapped = angle % TWO_PI;
  if (wrapped < 0.0) wrapped += TWO_PI;
  return wrapped;
}

function luminance(rgb0, rgb1, rgb2) {
  return 0.2126 * rgb0 + 0.7152 * rgb1 + 0.0722 * rgb2;
}

function skyTextureUvToLocalDir(u, v) {
  const azimuth = TWO_PI * u;
  const l = v * 2.0 - 1.0;
  const elevation = sign(l) * l * l * HALF_PI;
  const cosElevation = Math.cos(elevation);
  return [
    cosElevation * Math.cos(azimuth),
    cosElevation * Math.sin(azimuth),
    Math.sin(elevation),
  ];
}

function cyclesLocalAzimuth(dir) {
  return Math.atan2(dir[0], dir[1]);
}

function cyclesLocalElevation(dir) {
  return Math.asin(clamp(dir[2], -1.0, 1.0));
}

function cyclesSkyUvFromLocalDir(dir, sunRotation) {
  const dirElevation = cyclesLocalElevation(dir);
  return [
    fract((cyclesLocalAzimuth(dir) + PI + sunRotation) / TWO_PI),
    clamp(safeSqrt(Math.abs(dirElevation) / HALF_PI) * sign(dirElevation) * 0.5 + 0.5, 0.0, 1.0),
  ];
}

function sampleBilinearRgb(texture, width, height, u, v, outRgb) {
  const wrappedU = fract(u);
  const clampedV = clamp(v, 0.5 / height, 1.0 - 0.5 / height);
  const x = wrappedU * width - 0.5;
  const y = clampedV * height - 0.5;
  const xFloor = Math.floor(x);
  const yFloor = Math.floor(y);
  const x0 = ((xFloor % width) + width) % width;
  const x1 = (x0 + 1) % width;
  const y0 = clamp(yFloor, 0, height - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = x - xFloor;
  const fy = y - yFloor;

  const i00 = (y0 * width + x0) * 3;
  const i10 = (y0 * width + x1) * 3;
  const i01 = (y1 * width + x0) * 3;
  const i11 = (y1 * width + x1) * 3;

  for (let c = 0; c < 3; c++) {
    const c0 = mix(texture[i00 + c], texture[i10 + c], fx);
    const c1 = mix(texture[i01 + c], texture[i11 + c], fx);
    outRgb[c] = mix(c0, c1, fy);
  }
}

function simplifyMultiscatterAngles(sunElevation, sunRotation) {
  let newSunElevation = sunElevation;
  let newSunRotation = sunRotation;

  newSunElevation %= TWO_PI;
  if (Math.abs(newSunElevation) >= PI) {
    newSunElevation -= sign(newSunElevation) * TWO_PI;
  }
  if (newSunElevation >= HALF_PI || newSunElevation <= -HALF_PI) {
    newSunElevation = sign(newSunElevation) * PI - newSunElevation;
    newSunRotation += PI;
  }

  newSunRotation = wrapAnglePositive(newSunRotation);

  return {
    sunElevation: newSunElevation,
    sunRotation: wrapAnglePositive(newSunRotation),
  };
}

function raySphereIntersection(ox, oy, oz, dx, dy, dz, radius) {
  const b = ox * dx + oy * dy + oz * dz;
  const c = ox * ox + oy * oy + oz * oz - radius * radius;
  if (c > 0.0 && b > 0.0) return -1.0;
  const d = b * b - c;
  if (d < 0.0) return -1.0;
  return d >= b * b ? -b + Math.sqrt(d) : -b - Math.sqrt(d);
}

function xyzToLinearSrgb(x, y, z, out) {
  out[0] = Math.max(0.0, XYZ_TO_LINEAR_SRGB[0][0] * x + XYZ_TO_LINEAR_SRGB[0][1] * y + XYZ_TO_LINEAR_SRGB[0][2] * z);
  out[1] = Math.max(0.0, XYZ_TO_LINEAR_SRGB[1][0] * x + XYZ_TO_LINEAR_SRGB[1][1] * y + XYZ_TO_LINEAR_SRGB[1][2] * z);
  out[2] = Math.max(0.0, XYZ_TO_LINEAR_SRGB[2][0] * x + XYZ_TO_LINEAR_SRGB[2][1] * y + XYZ_TO_LINEAR_SRGB[2][2] * z);
}

function sunTransmittanceToXyz(transmittance, solidAngle, outXyz) {
  outXyz[0] =
    SPECTRAL_XYZ[0][0] * (SUN_SPECTRAL_IRRADIANCE[0] * transmittance[0] / solidAngle) +
    SPECTRAL_XYZ[1][0] * (SUN_SPECTRAL_IRRADIANCE[1] * transmittance[1] / solidAngle) +
    SPECTRAL_XYZ[2][0] * (SUN_SPECTRAL_IRRADIANCE[2] * transmittance[2] / solidAngle) +
    SPECTRAL_XYZ[3][0] * (SUN_SPECTRAL_IRRADIANCE[3] * transmittance[3] / solidAngle);
  outXyz[1] =
    SPECTRAL_XYZ[0][1] * (SUN_SPECTRAL_IRRADIANCE[0] * transmittance[0] / solidAngle) +
    SPECTRAL_XYZ[1][1] * (SUN_SPECTRAL_IRRADIANCE[1] * transmittance[1] / solidAngle) +
    SPECTRAL_XYZ[2][1] * (SUN_SPECTRAL_IRRADIANCE[2] * transmittance[2] / solidAngle) +
    SPECTRAL_XYZ[3][1] * (SUN_SPECTRAL_IRRADIANCE[3] * transmittance[3] / solidAngle);
  outXyz[2] =
    SPECTRAL_XYZ[0][2] * (SUN_SPECTRAL_IRRADIANCE[0] * transmittance[0] / solidAngle) +
    SPECTRAL_XYZ[1][2] * (SUN_SPECTRAL_IRRADIANCE[1] * transmittance[1] / solidAngle) +
    SPECTRAL_XYZ[2][2] * (SUN_SPECTRAL_IRRADIANCE[2] * transmittance[2] / solidAngle) +
    SPECTRAL_XYZ[3][2] * (SUN_SPECTRAL_IRRADIANCE[3] * transmittance[3] / solidAngle);
}

function sunDirectionFromElevation(elevation) {
  const z = Math.sin(elevation);
  const x = -safeSqrt(1.0 - z * z);
  return [x, 0.0, z];
}

function earthIntersectionAngle(altitudeMeters) {
  return HALF_PI - Math.asin(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitudeMeters / 1000.0));
}

class MultipleScatteringSky {
  constructor(airDensity, aerosolDensity, ozoneDensity) {
    this.airDensity = airDensity;
    this.aerosolDensity = aerosolDensity;
    this.ozoneDensity = ozoneDensity;
    this.transmittanceLut = new Float32Array(TRANSMITTANCE_RES_X * TRANSMITTANCE_RES_Y * 4);
    this.tmpA = new Float64Array(4);
    this.tmpB = new Float64Array(4);
    this.tmpC = new Float64Array(4);
    this.tmpD = new Float64Array(4);
    this.tmpE = new Float64Array(4);
    this.tmpF = new Float64Array(4);
    this.tmpG = new Float64Array(4);
    this.tmpH = new Float64Array(4);
  }

  _lutIndex(x, y) {
    return (y * TRANSMITTANCE_RES_X + x) * 4;
  }

  _getAtmosphereCollisionCoefficients(altitudeKm, aerosolAbsorption, aerosolScattering, molecularAbsorption, molecularScattering) {
    const localAerosolDensity = this.aerosolDensity * (
      AEROSOL_BASE_DENSITY * (Math.exp(-altitudeKm / AEROSOL_HEIGHT_SCALE) + (AEROSOL_BACKGROUND_DENSITY / AEROSOL_BASE_DENSITY))
    );
    const molecularScatteringScale = this.airDensity * Math.exp(-0.07771971 * Math.pow(Math.max(altitudeKm, 0.0), 1.16364243));
    const safeAltitude = Math.max(altitudeKm, 1e-4);
    const logAltitude = Math.log(safeAltitude);
    const ozoneDensity = this.ozoneDensity * OZONE_MEAN_DOBSON * 3.78547397e20 *
      Math.exp(-Math.pow(logAltitude - 3.22261, 2.0) * 5.55555555 - logAltitude);

    for (let i = 0; i < 4; i++) {
      aerosolAbsorption[i] = AEROSOL_ABSORPTION_CROSS_SECTION[i] * localAerosolDensity;
      aerosolScattering[i] = AEROSOL_SCATTERING_CROSS_SECTION[i] * localAerosolDensity;
      molecularAbsorption[i] = OZONE_ABSORPTION_CROSS_SECTION[i] * ozoneDensity;
      molecularScattering[i] = MOLECULAR_SCATTERING_COEFFICIENT_BASE[i] * molecularScatteringScale;
    }
  }

  precomputeTransmittanceLut() {
    for (let y = 0; y < TRANSMITTANCE_RES_Y; y++) {
      const v = y / (TRANSMITTANCE_RES_Y - 1);
      for (let x = 0; x < TRANSMITTANCE_RES_X; x++) {
        const u = x / (TRANSMITTANCE_RES_X - 1);
        this.computeTransmittance(u * 2.0 - 1.0, v, this.tmpA);
        const index = this._lutIndex(x, y);
        this.transmittanceLut[index + 0] = this.tmpA[0];
        this.transmittanceLut[index + 1] = this.tmpA[1];
        this.transmittanceLut[index + 2] = this.tmpA[2];
        this.transmittanceLut[index + 3] = this.tmpA[3];
      }
    }
  }

  computeTransmittance(cosTheta, normalizedAltitude, out) {
    const sunDx = -safeSqrt(1.0 - cosTheta * cosTheta);
    const sunDz = cosTheta;
    const distanceToEarthCenter = mix(EARTH_RADIUS_KM, ATMOSPHERE_RADIUS_KM, normalizedAltitude);
    const tMax = raySphereIntersection(0.0, 0.0, distanceToEarthCenter, sunDx, 0.0, sunDz, ATMOSPHERE_RADIUS_KM);
    const tStep = tMax / TRANSMITTANCE_STEPS;

    let r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;

    for (let step = 0; step < TRANSMITTANCE_STEPS; step++) {
      const t = (step + 0.5) * tStep;
      const px = sunDx * t;
      const pz = distanceToEarthCenter + sunDz * t;
      const altitudeKm = Math.max(Math.sqrt(px * px + pz * pz) - EARTH_RADIUS_KM, 0.0);

      this._getAtmosphereCollisionCoefficients(
        altitudeKm,
        this.tmpA,
        this.tmpB,
        this.tmpC,
        this.tmpD,
      );

      r0 += (this.tmpA[0] + this.tmpB[0] + this.tmpC[0] + this.tmpD[0]) * tStep;
      r1 += (this.tmpA[1] + this.tmpB[1] + this.tmpC[1] + this.tmpD[1]) * tStep;
      r2 += (this.tmpA[2] + this.tmpB[2] + this.tmpC[2] + this.tmpD[2]) * tStep;
      r3 += (this.tmpA[3] + this.tmpB[3] + this.tmpC[3] + this.tmpD[3]) * tStep;
    }

    out[0] = Math.exp(-r0);
    out[1] = Math.exp(-r1);
    out[2] = Math.exp(-r2);
    out[3] = Math.exp(-r3);
  }

  lookupTransmittance(cosTheta, normalizedAltitude, out) {
    const u = saturate(cosTheta * 0.5 + 0.5);
    const v = saturate(normalizedAltitude);
    const x = (TRANSMITTANCE_RES_X - 1) * u;
    const y = (TRANSMITTANCE_RES_Y - 1) * v;
    const x1 = Math.floor(x);
    const y1 = Math.floor(y);
    const x2 = Math.min(x1 + 1, TRANSMITTANCE_RES_X - 1);
    const y2 = Math.min(y1 + 1, TRANSMITTANCE_RES_Y - 1);
    const fx = x - x1;
    const fy = y - y1;

    const i11 = this._lutIndex(x1, y1);
    const i21 = this._lutIndex(x2, y1);
    const i12 = this._lutIndex(x1, y2);
    const i22 = this._lutIndex(x2, y2);

    for (let c = 0; c < 4; c++) {
      const bottom = mix(this.transmittanceLut[i11 + c], this.transmittanceLut[i21 + c], fx);
      const top = mix(this.transmittanceLut[i12 + c], this.transmittanceLut[i22 + c], fx);
      out[c] = mix(bottom, top, fy);
    }
  }

  lookupTransmittanceAtGround(cosTheta, out) {
    const u = saturate(cosTheta * 0.5 + 0.5);
    const x = (TRANSMITTANCE_RES_X - 1) * u;
    const x1 = Math.floor(x);
    const x2 = Math.min(x1 + 1, TRANSMITTANCE_RES_X - 1);
    const fx = x - x1;
    const i1 = this._lutIndex(x1, 0);
    const i2 = this._lutIndex(x2, 0);
    for (let c = 0; c < 4; c++) {
      out[c] = mix(this.transmittanceLut[i1 + c], this.transmittanceLut[i2 + c], fx);
    }
  }

  lookupTransmittanceToSun(normalizedAltitude, out) {
    const v = saturate(normalizedAltitude);
    const y = (TRANSMITTANCE_RES_Y - 1) * v;
    const y1 = Math.floor(y);
    const y2 = Math.min(y1 + 1, TRANSMITTANCE_RES_Y - 1);
    const fy = y - y1;
    const i1 = this._lutIndex(TRANSMITTANCE_RES_X - 1, y1);
    const i2 = this._lutIndex(TRANSMITTANCE_RES_X - 1, y2);
    for (let c = 0; c < 4; c++) {
      out[c] = mix(this.transmittanceLut[i1 + c], this.transmittanceLut[i2 + c], fy);
    }
  }

  lookupMultiscattering(cosTheta, normalizedHeight, distanceToEarthCenter, out) {
    const omega = TWO_PI * (1.0 - safeSqrt(1.0 - Math.pow(EARTH_RADIUS_KM / distanceToEarthCenter, 2.0)));

    this.lookupTransmittanceAtGround(cosTheta, this.tmpA);
    this.lookupTransmittanceToSun(0.0, this.tmpB);
    this.lookupTransmittanceToSun(normalizedHeight, this.tmpC);

    const groundFactor = PHASE_ISOTROPIC * omega * INV_PI * cosTheta;
    const atmosphereFactor = 0.02 * (1.0 / (1.0 + 5.0 * Math.exp(-17.92 * cosTheta)));

    for (let i = 0; i < 4; i++) {
      const ground = groundFactor * GROUND_ALBEDO[i] * this.tmpA[i] * (this.tmpB[i] / Math.max(this.tmpC[i], 1e-6));
      out[i] = ground + atmosphereFactor * MULTI_SCATTER_FIT[i];
    }
  }

  getInscattering(sunDir, rayOrigin, rayDir, tMax, outXyz) {
    const cosTheta = -(rayDir[0] * sunDir[0] + rayDir[1] * sunDir[1] + rayDir[2] * sunDir[2]);
    const molecularPhase = RAYLEIGH_PHASE_SCALE * (1.0 + cosTheta * cosTheta);
    const aerosolDen = 1.0 + AEROSOL_G2 + 2.0 * AEROSOL_G * cosTheta;
    const aerosolPhase = INV_FOUR_PI * (1.0 - AEROSOL_G2) / (aerosolDen * Math.sqrt(aerosolDen));
    const dt = tMax / IN_SCATTERING_STEPS;

    let l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;
    let t0 = 1.0, t1 = 1.0, t2 = 1.0, t3 = 1.0;

    for (let i = 0; i < IN_SCATTERING_STEPS; i++) {
      const t = (i + 0.5) * dt;
      const px = rayOrigin[0] + rayDir[0] * t;
      const py = rayOrigin[1] + rayDir[1] * t;
      const pz = rayOrigin[2] + rayDir[2] * t;
      const distanceToEarthCenter = Math.sqrt(px * px + py * py + pz * pz);
      const invDistance = 1.0 / Math.max(distanceToEarthCenter, 1e-6);
      const altitudeKm = Math.max(distanceToEarthCenter - EARTH_RADIUS_KM, 0.0);
      const normalizedAltitude = altitudeKm / ATMOSPHERE_THICKNESS_KM;
      const sampleCosTheta = (px * invDistance) * sunDir[0] + (py * invDistance) * sunDir[1] + (pz * invDistance) * sunDir[2];

      this._getAtmosphereCollisionCoefficients(
        altitudeKm,
        this.tmpA,
        this.tmpB,
        this.tmpC,
        this.tmpD,
      );
      this.lookupTransmittance(sampleCosTheta, normalizedAltitude, this.tmpE);
      this.lookupMultiscattering(sampleCosTheta, normalizedAltitude, distanceToEarthCenter, this.tmpF);

      const ext0 = this.tmpA[0] + this.tmpB[0] + this.tmpC[0] + this.tmpD[0];
      const ext1 = this.tmpA[1] + this.tmpB[1] + this.tmpC[1] + this.tmpD[1];
      const ext2 = this.tmpA[2] + this.tmpB[2] + this.tmpC[2] + this.tmpD[2];
      const ext3 = this.tmpA[3] + this.tmpB[3] + this.tmpC[3] + this.tmpD[3];

      const source0 = SUN_SPECTRAL_IRRADIANCE[0] * (
        this.tmpD[0] * (molecularPhase * this.tmpE[0] + this.tmpF[0]) +
        this.tmpB[0] * (aerosolPhase * this.tmpE[0] + this.tmpF[0])
      );
      const source1 = SUN_SPECTRAL_IRRADIANCE[1] * (
        this.tmpD[1] * (molecularPhase * this.tmpE[1] + this.tmpF[1]) +
        this.tmpB[1] * (aerosolPhase * this.tmpE[1] + this.tmpF[1])
      );
      const source2 = SUN_SPECTRAL_IRRADIANCE[2] * (
        this.tmpD[2] * (molecularPhase * this.tmpE[2] + this.tmpF[2]) +
        this.tmpB[2] * (aerosolPhase * this.tmpE[2] + this.tmpF[2])
      );
      const source3 = SUN_SPECTRAL_IRRADIANCE[3] * (
        this.tmpD[3] * (molecularPhase * this.tmpE[3] + this.tmpF[3]) +
        this.tmpB[3] * (aerosolPhase * this.tmpE[3] + this.tmpF[3])
      );

      const stepT0 = Math.exp(-dt * ext0);
      const stepT1 = Math.exp(-dt * ext1);
      const stepT2 = Math.exp(-dt * ext2);
      const stepT3 = Math.exp(-dt * ext3);

      l0 += t0 * ((source0 - source0 * stepT0) / Math.max(ext0, 1e-7));
      l1 += t1 * ((source1 - source1 * stepT1) / Math.max(ext1, 1e-7));
      l2 += t2 * ((source2 - source2 * stepT2) / Math.max(ext2, 1e-7));
      l3 += t3 * ((source3 - source3 * stepT3) / Math.max(ext3, 1e-7));

      t0 *= stepT0;
      t1 *= stepT1;
      t2 *= stepT2;
      t3 *= stepT3;
    }

    outXyz[0] = SPECTRAL_XYZ[0][0] * l0 + SPECTRAL_XYZ[1][0] * l1 + SPECTRAL_XYZ[2][0] * l2 + SPECTRAL_XYZ[3][0] * l3;
    outXyz[1] = SPECTRAL_XYZ[0][1] * l0 + SPECTRAL_XYZ[1][1] * l1 + SPECTRAL_XYZ[2][1] * l2 + SPECTRAL_XYZ[3][1] * l3;
    outXyz[2] = SPECTRAL_XYZ[0][2] * l0 + SPECTRAL_XYZ[1][2] * l1 + SPECTRAL_XYZ[2][2] * l2 + SPECTRAL_XYZ[3][2] * l3;
  }

  getSunDiscRgb(sunElevation, angularDiameter, altitudeMeters, outBottomRgb, outTopRgb) {
    const altitudeKm = clamp(altitudeMeters, 1.0, 99999.0) / 1000.0;
    const normalizedAltitude = altitudeKm / ATMOSPHERE_THICKNESS_KM;
    const halfAngular = angularDiameter * 0.5;
    const visibleFloor = -earthIntersectionAngle(altitudeMeters);
    const topElevation = sunElevation + halfAngular;
    if (topElevation <= visibleFloor) {
      outBottomRgb[0] = 0.0;
      outBottomRgb[1] = 0.0;
      outBottomRgb[2] = 0.0;
      outTopRgb[0] = 0.0;
      outTopRgb[1] = 0.0;
      outTopRgb[2] = 0.0;
      return;
    }

    const solidAngle = TWO_PI * (1.0 - Math.cos(halfAngular));
    const bottomElevation = Math.max(sunElevation - halfAngular, visibleFloor);
    const clampedTop = Math.max(topElevation, visibleFloor);

    this.computeTransmittance(Math.sin(bottomElevation), normalizedAltitude, this.tmpG);
    this.computeTransmittance(Math.sin(clampedTop), normalizedAltitude, this.tmpH);

    sunTransmittanceToXyz(this.tmpG, solidAngle, this.tmpC);
    sunTransmittanceToXyz(this.tmpH, solidAngle, this.tmpD);
    xyzToLinearSrgb(this.tmpC[0], this.tmpC[1], this.tmpC[2], outBottomRgb);
    xyzToLinearSrgb(this.tmpD[0], this.tmpD[1], this.tmpD[2], outTopRgb);
  }
}

function atmosphereKey(params) {
  return [
    params.airDensity.toFixed(3),
    params.aerosolDensity.toFixed(3),
    params.ozoneDensity.toFixed(3),
  ].join("|");
}

function skyTextureKey(params, simplified) {
  return [
    simplified.sunElevation.toFixed(6),
    params.altitude.toFixed(3),
  ].join("|");
}

function sunDataKey(params, simplified) {
  return [
    simplified.sunElevation.toFixed(6),
    params.altitude.toFixed(3),
    params.sunAngularDiameter.toFixed(6),
    params.sunIntensity.toFixed(3),
  ].join("|");
}

export class CyclesSkyModel {
  constructor(width, height, defaults = {}) {
    this.width = width;
    this.height = height;
    this.textureWidth = CYCLES_SKY_TEXTURE_WIDTH;
    this.textureHeight = CYCLES_SKY_TEXTURE_HEIGHT;
    this.defaults = { ...CYCLES_SKY_DEFAULTS, ...defaults };
    this.skyTextureRgb = new Float32Array(this.textureWidth * this.textureHeight * 3);
    this.packed = new Float32Array(width * height * 5 + height + 10);
    this.sunBottomRgb = new Float32Array(3);
    this.sunTopRgb = new Float32Array(3);
    this.sampleRgb = new Float32Array(3);
    this.lastAtmosphereKey = "";
    this.lastTextureKey = "";
    this.lastSunDataKey = "";
    this.lastRotationKey = "";
    this.earthIntersectionAngle = 0.0;
    this.sunAngularDiameter = this.defaults.sunAngularDiameter;
    this.sunSolidAngle = CYCLES_SUN_SOLID_ANGLE;
    this.atmosphereModel = null;
    this._nishitaModule = nishita;
    this.stats = {
      rebuiltAtmosphere: false,
      rebuiltTexture: false,
      rebuiltSunData: false,
      rebuiltPacked: false,
    };
  }

  _ensureAtmosphereModel(params) {
    const key = atmosphereKey(params);
    if (key === this.lastAtmosphereKey) {
      return false;
    }
    // Nishita port is stateless — no precomputation needed
    this.lastAtmosphereKey = key;
    this.lastTextureKey = "";
    this.lastSunDataKey = "";
    return true;
  }

  _recomputeSkyTexture(params, simplified) {
    // Use faithful Cycles Nishita port (sky-nishita.js)
    const { precomputeSkyTexture } = this._nishitaModule;
    const xyzPixels = precomputeSkyTexture(
      this.textureWidth, this.textureHeight,
      simplified.sunElevation,
      params.altitude,
      params.airDensity, params.aerosolDensity, params.ozoneDensity
    );

    // Convert XYZ → linear sRGB for the env map texture
    const rgb = [0, 0, 0];
    const n = this.textureWidth * this.textureHeight;
    let maxR=0, maxG=0, maxB=0, avgR=0, avgG=0, avgB=0;
    for (let i = 0; i < n; i++) {
      const X = xyzPixels[i * 4];
      const Y = xyzPixels[i * 4 + 1];
      const Z = xyzPixels[i * 4 + 2];
      xyzToLinearSrgb(X, Y, Z, rgb);
      this.skyTextureRgb[i * 3 + 0] = Math.max(0, rgb[0]);
      this.skyTextureRgb[i * 3 + 1] = Math.max(0, rgb[1]);
      this.skyTextureRgb[i * 3 + 2] = Math.max(0, rgb[2]);
      maxR = Math.max(maxR, rgb[0]); maxG = Math.max(maxG, rgb[1]); maxB = Math.max(maxB, rgb[2]);
      avgR += rgb[0]; avgG += rgb[1]; avgB += rgb[2];
    }
    console.log(`Nishita sky: max=(${maxR.toFixed(2)}, ${maxG.toFixed(2)}, ${maxB.toFixed(2)}) avg=(${(avgR/n).toFixed(3)}, ${(avgG/n).toFixed(3)}, ${(avgB/n).toFixed(3)})`);
  }

  _recomputeSunData(params, simplified) {
    // Use faithful Cycles Nishita port for sun disc
    const { precomputeSunDisc, earthIntersectionAngle: eia } = this._nishitaModule;
    const sun = precomputeSunDisc(
      simplified.sunElevation,
      params.sunAngularDiameter,
      params.altitude,
      params.airDensity, params.aerosolDensity
    );
    // Convert XYZ → RGB
    const brgb = [0,0,0], trgb = [0,0,0];
    xyzToLinearSrgb(sun.bottomXYZ[0], sun.bottomXYZ[1], sun.bottomXYZ[2], brgb);
    xyzToLinearSrgb(sun.topXYZ[0], sun.topXYZ[1], sun.topXYZ[2], trgb);
    // Scale sun disc: raw values are per-steradian (~3M), need ~1000-5000 for renderer
    const sunScale = params.sunIntensity * 0.0005;
    for (let c = 0; c < 3; c++) {
      this.sunBottomRgb[c] = Math.max(0, brgb[c]) * sunScale;
      this.sunTopRgb[c] = Math.max(0, trgb[c]) * sunScale;
    }
    this.earthIntersectionAngle = -eia(params.altitude);
    this.sunAngularDiameter = params.sunAngularDiameter;
    this.sunSolidAngle = TWO_PI * (1.0 - Math.cos(params.sunAngularDiameter * 0.5));
    console.log(`Sun disc: bottom=(${this.sunBottomRgb[0].toFixed(1)}, ${this.sunBottomRgb[1].toFixed(1)}, ${this.sunBottomRgb[2].toFixed(1)}) top=(${this.sunTopRgb[0].toFixed(1)}, ${this.sunTopRgb[1].toFixed(1)}, ${this.sunTopRgb[2].toFixed(1)})`);
  }

  _sampleCyclesSkyWorld(worldDir, sunRotation, outRgb) {
    const localDir = worldToCyclesLocal(worldDir);
    const [u, v] = cyclesSkyUvFromLocalDir(localDir, sunRotation);
    sampleBilinearRgb(this.skyTextureRgb, this.textureWidth, this.textureHeight, u, v, outRgb);
  }

  _rebuildPacked(params, simplified) {
    const pixelCount = this.width * this.height;
    const rgbOff = 0;
    const condOff = pixelCount * 3;
    const lumOff = condOff + pixelCount;
    const margOff = lumOff + pixelCount;
    const totalOff = margOff + this.height;
    const sunBottomOff = totalOff + 1;
    const sunTopOff = sunBottomOff + 3;
    const earthAngleOff = sunTopOff + 3;
    const angularDiameterOff = earthAngleOff + 1;
    const solidAngleOff = angularDiameterOff + 1;
    const rowSums = new Float64Array(this.height);

    for (let y = 0; y < this.height; y++) {
      const sinTheta = Math.max(Math.sin(((y + 0.5) / this.height) * PI), 1e-6);
      let rowSum = 0.0;
      for (let x = 0; x < this.width; x++) {
        const outPixel = y * this.width + x;
        const outRgb = outPixel * 3;
        const u = (x + 0.5) / this.width;
        const v = (y + 0.5) / this.height;
        const worldDir = envUvToWorldDir(u, v);
        this._sampleCyclesSkyWorld(worldDir, simplified.sunRotation, this.sampleRgb);

        this.packed[rgbOff + outRgb + 0] = this.sampleRgb[0];
        this.packed[rgbOff + outRgb + 1] = this.sampleRgb[1];
        this.packed[rgbOff + outRgb + 2] = this.sampleRgb[2];

        const importance = luminance(this.sampleRgb[0], this.sampleRgb[1], this.sampleRgb[2]) * sinTheta;
        this.packed[lumOff + outPixel] = importance;
        rowSum += importance;
      }

      rowSums[y] = rowSum;
      let cumulative = 0.0;
      const condBase = condOff + y * this.width;
      for (let x = 0; x < this.width; x++) {
        cumulative += this.packed[lumOff + y * this.width + x];
        this.packed[condBase + x] = rowSum > 0.0 ? cumulative / rowSum : (x + 1) / this.width;
      }
    }

    let grandTotal = 0.0;
    for (let y = 0; y < this.height; y++) grandTotal += rowSums[y];
    this.packed[totalOff] = grandTotal;

    let cumulativeRows = 0.0;
    for (let y = 0; y < this.height; y++) {
      cumulativeRows += rowSums[y];
      this.packed[margOff + y] = grandTotal > 0.0 ? cumulativeRows / grandTotal : (y + 1) / this.height;
    }

    this.packed[sunBottomOff + 0] = this.sunBottomRgb[0];
    this.packed[sunBottomOff + 1] = this.sunBottomRgb[1];
    this.packed[sunBottomOff + 2] = this.sunBottomRgb[2];
    this.packed[sunTopOff + 0] = this.sunTopRgb[0];
    this.packed[sunTopOff + 1] = this.sunTopRgb[1];
    this.packed[sunTopOff + 2] = this.sunTopRgb[2];
    this.packed[earthAngleOff] = this.earthIntersectionAngle;
    this.packed[angularDiameterOff] = this.sunAngularDiameter;
    this.packed[solidAngleOff] = this.sunSolidAngle;
  }

  update(overrides = {}) {
    const params = { ...this.defaults, ...overrides };
    const simplified = simplifyMultiscatterAngles(params.sunElevation, params.sunAzimuth);
    const textureKeyValue = skyTextureKey(params, simplified);
    const sunKeyValue = sunDataKey(params, simplified);
    const rotationKeyValue = simplified.sunRotation.toFixed(6);
    const rebuiltAtmosphere = this._ensureAtmosphereModel(params);
    let rebuiltTexture = false;
    let rebuiltSunData = false;
    let rebuiltPacked = false;

    if (textureKeyValue !== this.lastTextureKey) {
      this._recomputeSkyTexture(params, simplified);
      this.lastTextureKey = textureKeyValue;
      this.lastRotationKey = "";
      rebuiltTexture = true;
    }
    if (sunKeyValue !== this.lastSunDataKey) {
      this._recomputeSunData(params, simplified);
      this.lastSunDataKey = sunKeyValue;
      this.lastRotationKey = "";
      rebuiltSunData = true;
    }
    if (rebuiltTexture || rebuiltSunData || rotationKeyValue !== this.lastRotationKey) {
      this._rebuildPacked(params, simplified);
      this.lastRotationKey = rotationKeyValue;
      rebuiltPacked = true;
    }

    this.stats = { rebuiltAtmosphere, rebuiltTexture, rebuiltSunData, rebuiltPacked };
    return {
      buffer: this.packed,
      sunBottomRgb: this.sunBottomRgb,
      sunTopRgb: this.sunTopRgb,
      rebuiltAtmosphere,
      rebuiltTexture,
      rebuiltSunData,
      rebuiltPacked,
    };
  }
}
