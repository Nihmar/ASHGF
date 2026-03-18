use std::f64::consts::{E, PI};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Function {
    Sphere,
    Power,
    ExtendedRosenbrock,
    GeneralizedRosenbrock,
    ExtendedWhiteAndHolst,
    GeneralizedWhiteAndHolst,
    ExtendedFeudensteinAndRoth,
    ExtendedBaele,
    ExtendedPenalty,
    PerturbedQuadratic,
    AlmostPerturbedQuadratic,
    PerturbedQuadraticDiagonal,
    GeneralizedTridiagonal1,
    ExtendedTridiagonal1,
    Diagonal4,
    ExtendedHimmelblau,
    ExtendedPsc1,
    SinCos,
    ExtendedBd1,
    ExtendedMaratos,
    ExtendedCliff,
    ExtendedHiebert,
    QuadraticQf1,
    QuadraticQf2,
    ExtendedQuadraticPenaltyQp1,
    ExtendedQuadraticPenaltyQp2,
    ExtendedQuadraticExponentialEp1,
    ExtendedTridiagonal2,
    Fletchcr,
    Tridia,
    Arwhead,
    Nondia,
    Nondquar,
    Dqdrtic,
    BroydenTridiagonal,
    Liarwhd,
    Engval1,
    Edensch,
    Cube,
    Nonscomp,
    Vardim,
    Quartc,
    Sinquad,
    ExtendedDenschnb,
    ExtendedDenschnf,
    Dixon3dq,
    Cosine,
    Sine,
    Biggsb1,
    GeneralizedQuartic,
    Diagonal5,
    Diagonal7,
    Diagonal8,
    Fh3,
    Diagonal9,
    Himmelbg,
    Himmelh,
    Ackley,
    Griewank,
    Levy,
    Rastrigin,
    Schwefel,
    SumOfDifferentPowers,
    Trid,
    Zakharov,
    ExtendedTrigonometric,
    Eg2,
    Indef,
    Genhumps,
    Mccormck,
    Fletcbv3,
    Bdqrtic,
    Bdexp,
    Raydan1,
    Raydan2,
    Diagonal1,
    Diagonal2,
    Diagonal3,
    Hager,
}

impl Function {
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        match self {
            Function::Sphere => sphere(x),
            Function::Power => power(x),
            Function::ExtendedRosenbrock => extended_rosenbrock(x),
            Function::GeneralizedRosenbrock => generalized_rosenbrock(x),
            Function::ExtendedWhiteAndHolst => extended_white_and_holst(x),
            Function::GeneralizedWhiteAndHolst => generalized_white_and_holst(x),
            Function::ExtendedFeudensteinAndRoth => extended_feudenstein_and_roth(x),
            Function::ExtendedBaele => extended_baele(x),
            Function::ExtendedPenalty => extended_penalty(x),
            Function::PerturbedQuadratic => perturbed_quadratic(x),
            Function::AlmostPerturbedQuadratic => almost_perturbed_quadratic(x),
            Function::PerturbedQuadraticDiagonal => perturbed_quadratic_diagonal(x),
            Function::GeneralizedTridiagonal1 => generalized_tridiagonal_1(x),
            Function::ExtendedTridiagonal1 => extended_tridiagonal_1(x),
            Function::Diagonal4 => diagonal_4(x),
            Function::ExtendedHimmelblau => extended_himmelblau(x),
            Function::ExtendedPsc1 | Function::SinCos => extended_psc1(x),
            Function::ExtendedBd1 => extended_bd1(x),
            Function::ExtendedMaratos => extended_maratos(x),
            Function::ExtendedCliff => extended_cliff(x),
            Function::ExtendedHiebert => extended_hiebert(x),
            Function::QuadraticQf1 => quadratic_qf1(x),
            Function::QuadraticQf2 => quadratic_qf2(x),
            Function::ExtendedQuadraticPenaltyQp1 => extended_quadratic_penalty_qp1(x),
            Function::ExtendedQuadraticPenaltyQp2 => extended_quadratic_penalty_qp2(x),
            Function::ExtendedQuadraticExponentialEp1 => extended_quadratic_exponential_ep1(x),
            Function::ExtendedTridiagonal2 => extended_tridiagonal_2(x),
            Function::Fletchcr => fletchcr(x),
            Function::Tridia => tridia(x),
            Function::Arwhead => arwhead(x),
            Function::Nondia => nondia(x),
            Function::Nondquar => nondquar(x),
            Function::Dqdrtic => dqdrtic(x),
            Function::BroydenTridiagonal => broyden_tridiagonal(x),
            Function::Liarwhd => liarwhd(x),
            Function::Engval1 => engval1(x),
            Function::Edensch => edensch(x),
            Function::Cube => cube(x),
            Function::Nonscomp => nonscomp(x),
            Function::Vardim => vardim(x),
            Function::Quartc => quartc(x),
            Function::Sinquad => sinquad(x),
            Function::ExtendedDenschnb => extended_denschnb(x),
            Function::ExtendedDenschnf => extended_denschnf(x),
            Function::Dixon3dq => dixon3dq(x),
            Function::Cosine => cosine(x),
            Function::Sine => sine(x),
            Function::Biggsb1 => biggsb1(x),
            Function::GeneralizedQuartic => generalized_quartic(x),
            Function::Diagonal5 => diagonal_5(x),
            Function::Diagonal7 => diagonal_7(x),
            Function::Diagonal8 => diagonal_8(x),
            Function::Fh3 => fh3(x),
            Function::Diagonal9 => diagonal_9(x),
            Function::Himmelbg => himmelbg(x),
            Function::Himmelh => himmelh(x),
            Function::Ackley => ackley(x),
            Function::Griewank => griewank(x),
            Function::Levy => levy(x),
            Function::Rastrigin => rastrigin(x),
            Function::Schwefel => schwefel(x),
            Function::SumOfDifferentPowers => sum_of_different_powers(x),
            Function::Trid => trid(x),
            Function::Zakharov => zakharov(x),
            Function::ExtendedTrigonometric => extended_trigonometric(x),
            Function::Eg2 => eg2(x),
            Function::Indef => indef(x),
            Function::Genhumps => genhumps(x),
            Function::Mccormck => mccormck(x),
            Function::Fletcbv3 => fletcbv3(x),
            Function::Bdqrtic => bdqrtic(x),
            Function::Bdexp => bdexp(x),
            Function::Raydan1 => raydan_1(x),
            Function::Raydan2 => raydan_2(x),
            Function::Diagonal1 => diagonal_1(x),
            Function::Diagonal2 => diagonal_2(x),
            Function::Diagonal3 => diagonal_3(x),
            Function::Hager => hager(x),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Function::Sphere => "sphere",
            Function::Power => "power",
            Function::ExtendedRosenbrock => "extended_rosenbrock",
            Function::GeneralizedRosenbrock => "generalized_rosenbrock",
            Function::ExtendedWhiteAndHolst => "extended_white_and_holst",
            Function::GeneralizedWhiteAndHolst => "generalized_white_and_holst",
            Function::ExtendedFeudensteinAndRoth => "extended_feudenstein_and_roth",
            Function::ExtendedBaele => "extended_baele",
            Function::ExtendedPenalty => "extended_penalty",
            Function::PerturbedQuadratic => "perturbed_quadratic",
            Function::AlmostPerturbedQuadratic => "almost_perturbed_quadratic",
            Function::PerturbedQuadraticDiagonal => "perturbed_quadratic_diagonal",
            Function::GeneralizedTridiagonal1 => "generalized_tridiagonal_1",
            Function::ExtendedTridiagonal1 => "extended_tridiagonal_1",
            Function::Diagonal4 => "diagonal_4",
            Function::ExtendedHimmelblau => "extended_himmelblau",
            Function::ExtendedPsc1 => "extended_psc1",
            Function::SinCos => "sincos",
            Function::ExtendedBd1 => "extended_bd1",
            Function::ExtendedMaratos => "extended_maratos",
            Function::ExtendedCliff => "extended_cliff",
            Function::ExtendedHiebert => "extended_hiebert",
            Function::QuadraticQf1 => "quadratic_qf1",
            Function::QuadraticQf2 => "quadratic_qf2",
            Function::ExtendedQuadraticPenaltyQp1 => "extended_quadratic_penalty_qp1",
            Function::ExtendedQuadraticPenaltyQp2 => "extended_quadratic_penalty_qp2",
            Function::ExtendedQuadraticExponentialEp1 => "extended_quadratic_exponential_ep1",
            Function::ExtendedTridiagonal2 => "extended_tridiagonal_2",
            Function::Fletchcr => "fletchcr",
            Function::Tridia => "tridia",
            Function::Arwhead => "arwhead",
            Function::Nondia => "nondia",
            Function::Nondquar => "nondquar",
            Function::Dqdrtic => "dqdrtic",
            Function::BroydenTridiagonal => "broyden_tridiagonal",
            Function::Liarwhd => "liarwhd",
            Function::Engval1 => "engval1",
            Function::Edensch => "edensch",
            Function::Cube => "cube",
            Function::Nonscomp => "nonscomp",
            Function::Vardim => "vardim",
            Function::Quartc => "quartc",
            Function::Sinquad => "sinquad",
            Function::ExtendedDenschnb => "extended_denschnb",
            Function::ExtendedDenschnf => "extended_denschnf",
            Function::Dixon3dq => "dixon3dq",
            Function::Cosine => "cosine",
            Function::Sine => "sine",
            Function::Biggsb1 => "biggsb1",
            Function::GeneralizedQuartic => "generalized_quartic",
            Function::Diagonal5 => "diagonal_5",
            Function::Diagonal7 => "diagonal_7",
            Function::Diagonal8 => "diagonal_8",
            Function::Fh3 => "fh3",
            Function::Diagonal9 => "diagonal_9",
            Function::Himmelbg => "himmelbg",
            Function::Himmelh => "himmelh",
            Function::Ackley => "ackley",
            Function::Griewank => "griewank",
            Function::Levy => "levy",
            Function::Rastrigin => "rastrigin",
            Function::Schwefel => "schwefel",
            Function::SumOfDifferentPowers => "sum_of_different_powers",
            Function::Trid => "trid",
            Function::Zakharov => "zakharov",
            Function::ExtendedTrigonometric => "extended_trigonometric",
            Function::Eg2 => "eg2",
            Function::Indef => "indef",
            Function::Genhumps => "genhumps",
            Function::Mccormck => "mccormck",
            Function::Fletcbv3 => "fletcbv3",
            Function::Bdqrtic => "bdqrtic",
            Function::Bdexp => "bdexp",
            Function::Raydan1 => "raydan_1",
            Function::Raydan2 => "raydan_2",
            Function::Diagonal1 => "diagonal_1",
            Function::Diagonal2 => "diagonal_2",
            Function::Diagonal3 => "diagonal_3",
            Function::Hager => "hager",
        }
    }
}

// =============================================================================
// Benchmark Functions Implementation
// =============================================================================

#[inline]
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

#[inline]
pub fn power(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, xi)| (i as f64 + 1.0) * xi * xi)
        .sum()
}

#[inline]
pub fn extended_rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let a = x[i];
        let b = x[i + 1];
        let d = b - a * a;
        s += 100.0 * d * d + (1.0 - a) * (1.0 - a);
    }
    s
}

#[inline]
pub fn generalized_rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let d = x[i + 1] - x[i] * x[i];
        s += 100.0 * d * d + (1.0 - x[i]) * (1.0 - x[i]);
    }
    s
}

#[inline]
pub fn extended_white_and_holst(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let d = xd - xp * xp * xp;
        s += 100.0 * d * d + (1.0 - xp) * (1.0 - xp);
    }
    s
}

#[inline]
pub fn generalized_white_and_holst(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let d = x[i + 1] - x[i].powi(3);
        s += 100.0 * d * d + (1.0 - x[i]) * (1.0 - x[i]);
    }
    s
}

#[inline]
pub fn extended_feudenstein_and_roth(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let poly = ((5.0 - xd) * xd - 2.0) * xd;
        s += (-13.0 + xp + poly).powi(2) + (-29.0 + xp + poly).powi(2);
    }
    s
}

#[inline]
pub fn extended_baele(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        s += (1.5 - xp * (1.0 - xd)).powi(2);
        s += (2.25 - xp * (1.0 - xd * xd)).powi(2);
        s += (2.625 - xp * (1.0 - xd.powi(3))).powi(2);
    }
    s
}

#[inline]
pub fn extended_penalty(x: &[f64]) -> f64 {
    let mut s1 = 0.0;
    let mut sq = 0.0;
    let n = x.len();
    for xi in x.iter().take(n - 1) {
        let xi_val = *xi;
        let d = xi_val - 1.0;
        s1 += d * d;
        sq += xi_val * xi_val;
    }
    sq += x[n - 1] * x[n - 1];
    s1 + (sq - 0.25).powi(2)
}

#[inline]
pub fn perturbed_quadratic(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut total = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += (i as f64 + 1.0) * xi * xi;
        total += xi;
    }
    s + 0.01 * total * total
}

#[inline]
pub fn almost_perturbed_quadratic(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += (i as f64 + 1.0) * xi * xi;
    }
    s + 0.01 * (x[0] + x[x.len() - 1]).powi(2)
}

#[inline]
pub fn perturbed_quadratic_diagonal(x: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut sq_sum = 0.0;
    for (i, xi) in x.iter().enumerate() {
        total += xi;
        sq_sum += (i as f64 + 1.0) * xi * xi;
    }
    total * total + sq_sum / 100.0
}

#[inline]
pub fn generalized_tridiagonal_1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i] + x[i + 1] - 3.0;
        let b = x[i] - x[i + 1] + 1.0;
        s += a * a + b.powi(4);
    }
    s
}

#[inline]
pub fn extended_tridiagonal_1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let a = x[i] + x[i + 1] - 3.0;
        let b = x[i] - x[i + 1] + 1.0;
        s += a * a + b.powi(4);
    }
    s
}

#[inline]
pub fn diagonal_4(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        s += x[i] * x[i] + 100.0 * x[i + 1] * x[i + 1];
    }
    0.5 * s
}

#[inline]
pub fn extended_himmelblau(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let a = x[i] * x[i] + x[i + 1] - 11.0;
        let b = x[i] + x[i + 1] * x[i + 1] - 7.0;
        s += a * a + b * b;
    }
    s
}

#[inline]
pub fn extended_psc1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let quad = xp * xp + xd * xd + xp * xd;
        s += quad * quad + xp.sin().powi(2) + xd.cos().powi(2);
    }
    s
}

#[inline]
pub fn extended_bd1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let a = xp * xp + xd - 2.0;
        let b = (xp - 1.0).exp() - xp;
        s += a * a + b * b;
    }
    s
}

#[inline]
pub fn extended_maratos(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let d = xp * xp + xd * xd - 1.0;
        s += xp + 100.0 * d * d;
    }
    s
}

#[inline]
pub fn extended_cliff(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let diff = xp - xd;
        s += ((xp - 3.0) / 100.0).powi(2) + diff + (20.0 * diff).exp();
    }
    s
}

#[inline]
pub fn extended_hiebert(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        s += (xp - 10.0).powi(2) + (xp * xd - 50000.0).powi(2);
    }
    s
}

#[inline]
pub fn quadratic_qf1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += (i as f64 + 1.0) * xi * xi;
    }
    0.5 * s + x[x.len() - 1]
}

#[inline]
pub fn quadratic_qf2(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        let d = xi * xi - 1.0;
        s += (i as f64 + 1.0) * d * d;
    }
    0.5 * s + x[x.len() - 1]
}

#[inline]
pub fn extended_quadratic_penalty_qp1(x: &[f64]) -> f64 {
    let mut s1 = 0.0;
    let mut sq = 0.0;
    let n = x.len();
    for xi in x.iter().take(n - 1) {
        let xi_val = *xi;
        let d = xi_val * xi_val - 2.0;
        s1 += d * d;
        sq += xi_val * xi_val;
    }
    sq += x[n - 1] * x[n - 1];
    s1 + (sq - 0.5).powi(2)
}

#[inline]
pub fn extended_quadratic_penalty_qp2(x: &[f64]) -> f64 {
    let mut s1 = 0.0;
    let mut sq = 0.0;
    let n = x.len();
    for xi in x.iter().take(n - 1) {
        let xi_val = *xi;
        let d = xi_val * xi_val - xi_val.sin();
        s1 += d * d;
        sq += xi_val * xi_val;
    }
    sq += x[n - 1] * x[n - 1];
    s1 + (sq - 100.0).powi(2)
}

#[inline]
pub fn extended_quadratic_exponential_ep1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let d = x[i] - x[i + 1];
        let exp_d = d.exp();
        s += (exp_d - 5.0).powi(2) + d * d * (d - 11.0).powi(2);
    }
    s
}

#[inline]
pub fn extended_tridiagonal_2(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] * x[i] - 1.0;
        let b = x[i] + 1.0;
        s += a * a + 0.1 * b * b;
    }
    s
}

#[inline]
pub fn fletchcr(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let d = x[i + 1] - x[i] + 1.0 - x[i] * x[i];
        s += 100.0 * d * d;
    }
    s
}

#[inline]
pub fn tridia(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = (x[0] - 1.0).powi(2);
    for i in 1..n {
        let d = 2.0 * x[i] - x[i - 1];
        s += (i as f64 + 1.0) * d * d;
    }
    s
}

#[inline]
pub fn arwhead(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let x_last_sq = x[n - 1] * x[n - 1];
    for xi in x.iter().take(n - 1) {
        let xi_val = *xi;
        s1 += -4.0 * xi_val + 3.0;
        let d = xi_val * xi_val + x_last_sq;
        s2 += d * d;
    }
    s1 + s2
}

#[inline]
pub fn nondia(x: &[f64]) -> f64 {
    let n = x.len();
    let s = (x[0] - 1.0).powi(2);
    let mut total = 0.0;
    for i in 1..n {
        total += x[0] - x[i] * x[i];
    }
    s + 100.0 * total * total
}

#[inline]
pub fn nondquar(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = (x[0] - x[1]).powi(2);
    for i in 0..n - 2 {
        let d = x[i] + x[i + 1] + x[n - 1];
        s += d.powi(4);
    }
    s += (x[n - 2] + x[n - 1]).powi(2);
    s
}

#[inline]
pub fn dqdrtic(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in 0..n - 2 {
        s += x[i].powi(2) + 100.0 * x[i + 1].powi(2) + 100.0 * x[i + 2].powi(2);
    }
    s
}

#[inline]
pub fn broyden_tridiagonal(x: &[f64]) -> f64 {
    let n = x.len();
    let a = x[0];
    let a2 = a * a;
    let mut s = (3.0 * a - 2.0 * a2).powi(2);
    for i in 1..n - 1 {
        let xi = x[i];
        let xi2 = xi * xi;
        let d = 3.0 * xi - 2.0 * xi2 - x[i - 1] - 2.0 * x[i + 1] + 1.0;
        s += d * d;
    }
    let b = x[n - 1];
    let b2 = b * b;
    s += (3.0 * b - 2.0 * b2 - x[n - 2] + 1.0).powi(2);
    s
}

#[inline]
pub fn liarwhd(x: &[f64]) -> f64 {
    let x0 = x[0];
    let mut s = 0.0;
    for xi in x {
        s += 4.0 * (xi * xi - x0).powi(2) + (xi - 1.0).powi(2);
    }
    s
}

#[inline]
pub fn engval1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let d = x[i] * x[i] + x[i + 1] * x[i + 1];
        s += d * d - 4.0 * x[i] + 3.0;
    }
    s
}

#[inline]
pub fn edensch(x: &[f64]) -> f64 {
    let mut s = 16.0;
    for i in 0..x.len() - 1 {
        let xi = x[i];
        let xi1 = x[i + 1];
        s += (xi - 2.0).powi(4) + (xi * xi1 + 2.0 * xi1).powi(2) + (xi1 + 1.0).powi(2);
    }
    s
}

#[inline]
pub fn cube(x: &[f64]) -> f64 {
    let mut s = (x[0] - 1.0).powi(2);
    for i in 1..x.len() {
        let d = x[i] - x[i - 1].powi(3);
        s += 100.0 * d * d;
    }
    s
}

#[inline]
pub fn nonscomp(x: &[f64]) -> f64 {
    let mut s = (x[0] - 1.0).powi(2);
    for i in 1..x.len() {
        let d = x[i] - x[i - 1].powi(2);
        s += 4.0 * d * d;
    }
    s
}

#[inline]
pub fn vardim(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s1 += (xi - 1.0).powi(2);
        s2 += (i as f64 + 1.0) * xi;
    }
    let t = s2 - n * (n + 1.0) / 2.0;
    s1 + t * t + t.powi(4)
}

#[inline]
pub fn quartc(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for xi in x {
        let d = xi - 1.0;
        s += d.powi(4);
    }
    s
}

#[inline]
pub fn sinquad(x: &[f64]) -> f64 {
    let n = x.len();
    let x0sq = x[0] * x[0];
    let xnsq = x[n - 1] * x[n - 1];
    let mut s = (x[0] - 1.0).powi(4);
    for i in 1..n - 1 {
        let xisq = x[i] * x[i];
        let d = (x[i] - x[n - 1]).sin() - x0sq + xisq;
        s += d * d;
    }
    s += (xnsq - x0sq).powi(2);
    s
}

#[inline]
pub fn extended_denschnb(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let d = (xp - 2.0).powi(2);
        s += d + d * xd * xd + (xd + 1.0).powi(2);
    }
    s
}

#[inline]
pub fn extended_denschnf(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        let t1 = 2.0 * (xp + xd).powi(2) + (xp - xd).powi(2) - 8.0;
        s += t1 * t1 + (5.0 * xp * xp + (xp - 3.0).powi(2) - 9.0).powi(2);
    }
    s
}

#[inline]
pub fn dixon3dq(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = (x[0] - 1.0).powi(2) + (x[n - 1] - 1.0).powi(2);
    for i in 0..n - 1 {
        let d = x[i] - x[i + 1];
        s += d * d;
    }
    s
}

#[inline]
pub fn cosine(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += (-0.5 * x[i + 1] + x[i] * x[i]).cos();
    }
    s
}

#[inline]
pub fn sine(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += (-0.5 * x[i + 1] + x[i] * x[i]).sin();
    }
    s
}

#[inline]
pub fn biggsb1(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = (x[0] - 1.0).powi(2) + (x[n - 1] - 1.0).powi(2);
    for i in 0..n - 1 {
        let d = x[i + 1] - x[i];
        s += d * d;
    }
    s
}

#[inline]
pub fn generalized_quartic(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let xi2 = x[i] * x[i];
        let d = x[i + 1] + xi2;
        s += xi2 + d * d;
    }
    s
}

#[inline]
pub fn diagonal_5(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for xi in x {
        s += (xi.exp() + (-xi).exp()).ln();
    }
    s
}

#[inline]
pub fn diagonal_7(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for xi in x {
        s += xi.exp() - 2.0 * xi - xi * xi;
    }
    s
}

#[inline]
pub fn diagonal_8(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for xi in x {
        s += xi * xi.exp() - 2.0 * xi - xi * xi;
    }
    s
}

#[inline]
pub fn fh3(x: &[f64]) -> f64 {
    let mut total = 0.0;
    for xi in x {
        total += xi;
    }
    let s = total * total;
    let mut s2 = 0.0;
    for xi in x {
        s2 += xi * xi.exp() - 2.0 * xi - xi * xi;
    }
    s + s2
}

#[inline]
pub fn diagonal_9(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += xi.exp() - (i as f64 + 1.0) * xi;
    }
    s + 10000.0 * x[x.len() - 1] * x[x.len() - 1]
}

#[inline]
pub fn himmelbg(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        s += (2.0 * xp * xp + 3.0 * xd * xd) * (-xp - xd).exp();
    }
    s
}

#[inline]
pub fn himmelh(x: &[f64]) -> f64 {
    let mut s = 0.0;
    let n = x.len();
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        s += -3.0 * xp - 2.0 * xd + 2.0 + xp.powi(3) + xd * xd;
    }
    s
}

#[inline]
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut sq_sum = 0.0;
    let mut cos_sum = 0.0;
    let c = 2.0 * PI;
    for xi in x {
        sq_sum += xi * xi;
        cos_sum += (c * xi).cos();
    }
    let t1 = -20.0 * (-0.2 * (sq_sum / n).sqrt()).exp();
    let t2 = -(cos_sum / n).exp();
    t1 + t2 + 20.0 + E
}

#[inline]
pub fn griewank(x: &[f64]) -> f64 {
    let mut sq = 0.0;
    let mut prod = 1.0;
    for (i, xi) in x.iter().enumerate() {
        sq += xi * xi;
        prod *= (xi / (i as f64 + 1.0).sqrt()).cos();
    }
    sq / 4000.0 - prod + 1.0
}

#[inline]
pub fn levy(x: &[f64]) -> f64 {
    let n = x.len();
    let pi = PI;

    let w = |xi: f64| -> f64 { 1.0 + (xi - 1.0) / 4.0 };

    let w0 = w(x[0]);
    let mut s = (pi * w0).sin().powi(2);
    for xi in x.iter().take(n - 1) {
        let wi = w(*xi);
        s += (wi - 1.0).powi(2) * (1.0 + 10.0 * (pi * wi + 1.0).sin().powi(2));
    }
    let wd = w(x[n - 1]);
    s += (wd - 1.0).powi(2) * (1.0 + (2.0 * pi * wd).sin().powi(2));
    s
}

#[inline]
pub fn rastrigin(x: &[f64]) -> f64 {
    let c = 2.0 * PI;
    let mut s = 10.0 * x.len() as f64;
    for xi in x {
        s += xi * xi - 10.0 * (c * xi).cos();
    }
    s
}

#[inline]
pub fn schwefel(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for xi in x {
        s += xi * (xi.abs().sqrt()).sin();
    }
    418.9829 * x.len() as f64 - s
}

#[inline]
pub fn sum_of_different_powers(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += xi.abs().powi(i as i32 + 2);
    }
    s
}

#[inline]
pub fn trid(x: &[f64]) -> f64 {
    let mut s1 = 0.0;
    for xi in x {
        s1 += (xi - 1.0).powi(2);
    }
    let mut s2 = 0.0;
    for i in 1..x.len() {
        s2 += x[i] * x[i - 1];
    }
    s1 - s2
}

#[inline]
pub fn zakharov(x: &[f64]) -> f64 {
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s1 += xi * xi;
        s2 += 0.5 * (i as f64 + 1.0) * xi;
    }
    s1 + s2 * s2 + s2.powi(4)
}

#[inline]
pub fn extended_trigonometric(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut cos_sum = 0.0;
    for xi in x {
        cos_sum += xi.cos();
    }
    let base = n - cos_sum;
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        let term = base + (i as f64 + 1.0) * (1.0 - xi.cos()) + xi.sin();
        s += term * term;
    }
    s
}

#[inline]
pub fn eg2(x: &[f64]) -> f64 {
    let x0 = x[0];
    let mut s = 0.0;
    let n = x.len();
    for xi in x.iter().take(n - 1) {
        s += (x0 + xi * xi - 1.0).sin();
    }
    s += 0.5 * (x[n - 1] * x[n - 1]).sin();
    s
}

#[inline]
pub fn indef(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = 0.0;
    for xi in x {
        s += xi;
    }
    let x0 = x[0];
    let xn = x[n - 1];
    let mut t = 0.0;
    for xi in x.iter().skip(1).take(n - 2) {
        t += (2.0 * xi - xn - x0).cos();
    }
    s + 0.5 * t
}

#[inline]
pub fn genhumps(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += (2.0 * x[i]).sin().powi(2) * (2.0 * x[i + 1]).sin().powi(2)
            + 0.05 * (x[i].powi(2) + x[i + 1].powi(2));
    }
    s
}

#[inline]
pub fn mccormck(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i];
        let b = x[i + 1];
        s += -1.5 * a + 2.5 * b + 1.0 + (a - b).powi(2) + (a + b).sin();
    }
    s
}

#[inline]
pub fn fletcbv3(x: &[f64]) -> f64 {
    let n = x.len();
    let p = 1e-8;
    let h = 1.0 / (n as f64 + 1.0);
    let factor = p * (h * h + 2.0) / (h * h);
    let c_over_h2 = p / (h * h);

    let mut s = 0.5 * p * (x[0].powi(2) + x[n - 1].powi(2));

    for i in 0..n - 1 {
        let diff = x[i] - x[i + 1];
        s += 0.5 * p * diff * diff;
    }

    s += factor * x.iter().sum::<f64>();
    s += c_over_h2 * x.iter().map(|xi| xi.cos()).sum::<f64>();

    s
}

#[inline]
#[allow(clippy::needless_range_loop)]
pub fn bdqrtic(x: &[f64]) -> f64 {
    let n = x.len();
    let mut term_1 = 0.0;
    for i in 0..n - 3 {
        term_1 += (-4.0 * x[i] + 3.0).powi(2);
    }

    let mut term_2 = 0.0;
    for i in 0..n - 3 {
        let x0 = x[i];
        let x1 = x[i + 1];
        let x2 = x[i + 2];
        let x3 = x[i + 3];
        let term =
            x0 * x0 + 2.0 * x1 * x1 + 3.0 * x2 * x2 + 4.0 * x3 * x3 + 5.0 * x[n - 1] * x[n - 1];
        term_2 += term * term;
    }

    term_1 + term_2
}

#[inline]
pub fn bdexp(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 2 {
        let t = x[i] + x[i + 1];
        s += t * (-x[i + 2] * t).exp();
    }
    s
}

#[inline]
pub fn raydan_1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += (i as f64 + 1.0) * (xi.exp() - xi);
    }
    0.1 * s
}

#[inline]
pub fn raydan_2(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for xi in x {
        s += xi.exp() - xi;
    }
    s
}

#[inline]
pub fn diagonal_1(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += xi.exp() - (i as f64 + 1.0) * xi;
    }
    s
}

#[inline]
pub fn diagonal_2(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += xi.exp() - xi / (i as f64 + 1.0);
    }
    s
}

#[inline]
pub fn diagonal_3(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += xi.exp() - (i as f64 + 1.0) * xi.sin();
    }
    s
}

#[inline]
pub fn hager(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, xi) in x.iter().enumerate() {
        s += xi.exp() - ((i as f64 + 1.0).sqrt()) * xi;
    }
    s
}

pub trait BenchmarkFunction: Fn(&[f64]) -> f64 + Copy {
    fn name(&self) -> &'static str;
}

#[allow(dead_code)]
pub struct FunctionWrapper {
    func: fn(&[f64]) -> f64,
    name: &'static str,
}

impl FunctionWrapper {
    pub fn new(func: fn(&[f64]) -> f64, name: &'static str) -> Self {
        Self { func, name }
    }

    pub fn evaluate(&self, x: &[f64]) -> f64 {
        (self.func)(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere() {
        let x = vec![1.0, 2.0, 3.0];
        approx::assert_abs_diff_eq!(sphere(&x), 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_at_zero() {
        let x = vec![0.0, 0.0, 0.0];
        approx::assert_abs_diff_eq!(sphere(&x), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ackley_at_origin() {
        let x = vec![0.0; 10];
        let result = ackley(&x);
        approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rastrigin_at_origin() {
        let x = vec![0.0; 10];
        let result = rastrigin(&x);
        approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_at_minimum() {
        let x = vec![1.0; 10];
        let result = generalized_rosenbrock(&x);
        approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }
}
