"""Explicit CPU-only LQR preparation: inverse dynamics, native FD, bounded rollouts.

This script DOES execute physics; importing it does not. It is run by the USER,
once before stand-lqr training. No rendering, GPU jobs or dependency installation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares

from lqr_artifact import DEFAULT_FILE, KEEP, NQ, NV, NU, SCHEMA, source_digest

TARGET_EPS = 1e-8  # Only roundoff may be projected, never a real target violation.
TARGET_GUARD = 1e-5  # Search inside the target bounds instead of on their kink.


def candidate_diagnostics(root, torque_ratio, ctrl, lower, upper, weight, names):
    """Pure numeric constraints, also used to rank ALL multi-start candidates."""
    force = float(np.linalg.norm(root[:3]))
    moment = float(np.linalg.norm(root[3:]))
    over = np.maximum(lower-ctrl,0.)+np.maximum(ctrl-upper,0.)
    maximum = float(np.max(over))
    finite = bool(np.isfinite(root).all() and np.isfinite(ctrl).all() and np.isfinite(torque_ratio))
    reasons = []
    if not finite: reasons.append('nonfinite state/control')
    if force > .01*weight: reasons.append(f'root force {force:.6g}N > {.01*weight:.6g}N')
    if moment > .01*weight: reasons.append(f'root moment {moment:.6g}Nm > {.01*weight:.6g}Nm')
    if torque_ratio > .9: reasons.append(f'torque fraction {torque_ratio:.6g} > 0.9')
    for i in np.flatnonzero(over>TARGET_EPS):
        reasons.append(f'{names[i]} target={ctrl[i]:.12g}rad outside '
                       f'[{lower[i]:.12g}, {upper[i]:.12g}], violation={over[i]:.6g}rad')
    violation = max(force/(.01*weight)-1.,moment/(.01*weight)-1.,
                    torque_ratio/.9-1.,(maximum-TARGET_EPS)/.05,0.) if finite else float('inf')
    return {'feasible':not reasons,'reasons':reasons,'max_target_violation_rad':maximum,
            'rank':(bool(reasons),violation,(force+moment)/weight)}


def make_model():
    from balance_task import robot_spec, terrain_stand_cfg
    spec = robot_spec()
    spec.option.gravity = [0., 0., -9.81/6]
    spec.option.timestep = 1/120
    spec.worldbody.add_geom(name='lqr_ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[5., 5., .1], contype=1, conaffinity=1,
                           friction=[1., .01, .001])
    model = spec.compile()
    terrain_stand_cfg().sim.mujoco.apply(model)
    if (model.nq, model.nv, model.nu, model.na) != (NQ, NV, NU, 0):
        raise ValueError('Expected the native 23-axis position-PD model, with no activation state')
    return model


def equilibrium(model):
    """Find small pose changes with negligible *all-six* floating-root residual.

    Inverse dynamics includes native soft contacts; no root force is applied.
    Source FBX pose remains the regularization reference, not an assumed equilibrium.
    """
    data = mujoco.MjData(model)
    qsrc = model.qpos0.copy()
    jids = model.actuator_trnid[:, 0].astype(int)
    qadr, vadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    kp = model.actuator_gainprm[:, 0]
    limits = model.actuator_forcerange[:, 1]
    target_lo = np.maximum(model.jnt_range[jids,0],-.5)
    target_hi = np.minimum(model.jnt_range[jids,1],.5)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    mg = float(model.body_subtreemass[1] * abs(model.opt.gravity[2]))
    # Optimization variables: root height/roll/pitch and 23 joint offsets.
    lo = np.r_[qsrc[2]-.08, -.14, -.14, np.maximum(model.jnt_range[1:, 0], -.3)]
    hi = np.r_[qsrc[2]+.08, .14, .14, np.minimum(model.jnt_range[1:, 1], .3)]

    def assign(z):
        data.qpos[:] = qsrc
        data.qpos[2] = z[0]
        rotation = np.array([z[1], z[2], 0.])
        mujoco.mju_quatIntegrate(data.qpos[3:7], rotation, 1.)
        data.qpos[7:] = z[3:]
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.
        mujoco.mj_inverse(model, data)
        ctrl = data.qpos[qadr] + data.qfrc_inverse[vadr]/kp
        return ctrl

    def residual(z):
        ctrl = assign(z)
        torque = data.qfrc_inverse[vadr]
        # Force in N and torque in Nm normalized separately (1m length scale).
        root = data.qfrc_inverse[:6]/mg
        torque_over = np.maximum(np.abs(torque)/limits-.85, 0.)
        target_over = (np.maximum(target_lo+TARGET_GUARD-ctrl,0.)
                       +np.maximum(ctrl-(target_hi-TARGET_GUARD),0.))/.05
        return np.r_[root, torque_over, target_over,
                     .0002*z[1:3], .0002*z[3:]]

    best = None
    for height in (-.0005, -.002, .001):
        seed = np.clip(np.r_[qsrc[2]+height, 0., 0., qsrc[7:]], lo+1e-9, hi-1e-9)
        fit = least_squares(residual, seed, bounds=(lo, hi),
                            max_nfev=400, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        candidate_ctrl = assign(fit.x)
        candidate = candidate_diagnostics(data.qfrc_inverse[:6],
            float(np.max(np.abs(data.qfrc_inverse[vadr])/limits)),candidate_ctrl,
            target_lo,target_hi,mg,actuator_names)
        print(f'[LQR candidate] seed_dz={height:g}m, feasible={candidate["feasible"]}, '
              f'max_target_violation={candidate["max_target_violation_rad"]:.6g}rad; '
              f'{"; ".join(candidate["reasons"]) or "all inverse-dynamics constraints passed"}',flush=True)
        if best is None or candidate['rank'] < best[0]:
            best = candidate['rank'], fit.x.copy()
    ctrl = assign(best[1])
    q0, inverse_root = data.qpos.copy(), data.qfrc_inverse[:6].copy()
    torque_ratio = float(np.max(np.abs(data.qfrc_inverse[vadr])/limits))
    candidate = candidate_diagnostics(inverse_root,torque_ratio,ctrl,target_lo,target_hi,mg,actuator_names)
    if not candidate['feasible']:
        raise RuntimeError('No feasible local equilibrium: '+'; '.join(candidate['reasons'])+
                           f'; root residual={inverse_root}. No gain was saved.')
    # Only <=1e-8rad roundoff can reach this projection. The ACTUAL projected
    # PD target still must pass the forward and closed-loop checks below.
    correction = float(np.max(np.abs(np.clip(ctrl,target_lo,target_hi)-ctrl)))
    ctrl = np.clip(ctrl,target_lo,target_hi)
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    loads = [float(data.sensor('balance_'+side+'_load').data[0]) for side in ('left', 'right')]
    if min(loads) < .12*mg or np.max(np.abs(data.qacc)) > .5:
        raise RuntimeError(f'Forward check failed: loads={loads}, max acceleration='
                           f'{np.max(np.abs(data.qacc)):.4g}. No gain was saved.')
    return q0, ctrl, {'root_inverse_force_N': inverse_root[:3].tolist(),
                      'root_inverse_torque_Nm': inverse_root[3:].tolist(),
                      'max_torque_fraction': torque_ratio, 'foot_normal_load_N': loads,
                      'target_roundoff_projection_rad':correction,
                      'max_abs_forward_qacc': float(np.max(np.abs(data.qacc)))}


def gains(model, q0, ctrl):
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    A = np.empty((2*NV, 2*NV)); B = np.empty((2*NV, NU))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    # Native free-root rotation differences/qvel use body coordinates. Convert
    # both to the yaw-local world basis before removing the WORLD-yaw gauge.
    rotation=np.empty(9)
    mujoco.mju_quat2Mat(rotation,q0[3:7])
    rotation=rotation.reshape(3,3)
    transform=np.eye(2*NV)
    transform[3:6,3:6]=rotation
    transform[NV+3:NV+6,NV+3:NV+6]=rotation
    A=transform@A@transform.T
    B=transform@B
    # The residual action and LQR base are held across BOTH 120Hz physics steps.
    B = A @ B + B
    A = A @ A
    dropped = np.setdiff1d(np.arange(2*NV), KEEP)
    gauge_coupling = float(np.max(np.abs(A[np.ix_(KEEP, dropped)])))
    if gauge_coupling > 1e-3:
        raise RuntimeError(f'Translation/yaw reduction invalid: coupling={gauge_coupling}')
    jac_com = np.zeros((3, NV))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body('Pelvis').id)
    jac_feet = np.zeros((3, NV))
    for side in ('left', 'right'):
        j = np.zeros((3, NV))
        mujoco.mj_jacSite(model, data, j, None, model.site('balance_'+side+'_sole').id)
        jac_feet += .5*j
    J = (jac_com-jac_feet)[:2]@transform[:NV,:NV].T
    qjoint = np.zeros(NV)
    for i in range(1, model.njnt):
        name = model.joint(i).name
        qjoint[model.jnt_dofadr[i]] = 3. if any(k in name for k in ('Hip','Knee','Ankle','Waist')) else .3
    Qpos = 1000.*J.T@J + np.diag(qjoint)
    Qpos[2, 2] += 20.
    Q = np.zeros_like(A)
    Q[:NV, :NV] = Qpos
    Q[NV:, NV:] = np.diag(np.r_[np.ones(3)*.5, np.ones(3)*.1, np.ones(NU)*.02])
    Ar, Br, Qr = A[np.ix_(KEEP,KEEP)], B[KEEP], Q[np.ix_(KEEP,KEEP)]
    R = 2.*np.eye(NU)
    P = solve_discrete_are(Ar, Br, Qr, R)
    K = np.linalg.solve(R+Br.T@P@Br, Br.T@P@Ar)
    radius = float(np.max(np.abs(np.linalg.eigvals(Ar-Br@K))))
    if not np.isfinite(K).all() or radius >= 1.-1e-7:
        raise RuntimeError(f'LQR reduced closed-loop not stable: radius={radius}')
    return K, {'spectral_radius': radius, 'gauge_coupling': gauge_coupling}


def rollout_motion_failures(tail_motion):
    """Judge the REAL limited closed loop, not whether clipping occurred."""
    limits = (.05,.1,.1)
    labels = ('root linear speed (m/s)','root angular speed (rad/s)','joint speed RMS (rad/s)')
    return [f'{label}={value:.6g}, limit={limit:g}'
            for value,limit,label in zip(tail_motion,limits,labels)
            if not np.isfinite(value) or value>limit]


def validate(model, q0, ctrl, K, seconds):
    """Bounded CPU flat-ground smoke checks, not terrain/performance evidence."""
    import torch
    from lqr_action import blend_commands
    jids=model.actuator_trnid[:,0].astype(int)
    lo=torch.tensor(np.maximum(model.jnt_range[jids,0],-.5))
    hi=torch.tensor(np.minimum(model.jnt_range[jids,1],.5))
    gain=torch.tensor(K); reference=torch.tensor(ctrl)
    source=torch.tensor(model.qpos0[model.jnt_qposadr[jids]])[None,:]
    zero=torch.zeros((1,NU),dtype=torch.float64)
    mg=float(model.body_subtreemass[1]*abs(model.opt.gravity[2]))
    reports = []
    for label, roll, speed in (('rest',0.,0.), ('roll_plus',.01,0.),
                               ('roll_minus',-.01,0.), ('velocity',0.,.02),
                               ('source_spawn',0.,0.)):
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0 if label=='source_spawn' else q0
        if label=='source_spawn':
            data.qpos[2]+=.015  # Match the actual source-pose flat reset gap.
        mujoco.mju_quatIntegrate(data.qpos[3:7], np.array([roll,0.,0.]), 1.)
        data.qvel[0] = speed
        dq = np.zeros(NV)
        peak = 0.; saturated = 0; last_motion = []
        max_clip = 0.; clipped_joints = np.zeros(NU,dtype=int)
        for step in range(round(seconds/(2*model.opt.timestep))):
            mujoco.mj_forward(model,data)
            mujoco.mj_differentiatePos(model, dq, 1., q0, data.qpos)
            full_state=np.r_[dq,data.qvel]
            mujoco.mju_rotVecQuat(full_state[3:6],dq[3:6],q0[3:7])
            mujoco.mju_rotVecQuat(full_state[NV+3:NV+6],data.qvel[3:6],data.qpos[3:7])
            dx = full_state[KEEP]
            load=[float(data.sensor('balance_'+side+'_load').data[0])/mg for side in ('left','right')]
            _,command,alpha,clipped=blend_commands(torch.tensor(dx)[None,:],gain,reference,source,
                torch.tensor([load]),torch.zeros(1),zero,lo,hi,.05)
            saturated += int(float(clipped[0])>0)
            requested=source.numpy()[0]+float(alpha[0])*(ctrl-K@dx-source.numpy()[0])
            clip_gap=np.abs(requested-command[0].numpy())
            max_clip=max(max_clip,float(clip_gap.max()))
            clipped_joints+=(clip_gap>1e-6)
            data.ctrl[:] = command[0].numpy()
            for _ in range(2):
                mujoco.mj_step(model, data)
            mujoco.mj_differentiatePos(model, dq, 1., q0, data.qpos)
            spatial_rotation=np.empty(3)
            mujoco.mju_rotVecQuat(spatial_rotation,dq[3:6],q0[3:7])
            angle = float(np.linalg.norm(spatial_rotation[:2])); peak = max(peak, angle)
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or angle > .25
                    or data.qpos[2] < .8*q0[2]):
                raise RuntimeError(f'LQR {label} smoke check fell at {data.time:.3f}s; no artifact saved')
            if step > .8*seconds/(2*model.opt.timestep):
                last_motion.append([np.linalg.norm(data.qvel[:3]),np.linalg.norm(data.qvel[3:6]),
                                    np.sqrt(np.mean(data.qvel[6:]**2))])
        tail_motion=np.mean(last_motion,axis=0)
        report={'case':label,'seconds':seconds,'peak_tilt_error_rad':peak,
                'tail_root_speed_m_s':float(tail_motion[0]),
                'tail_root_angular_speed_rad_s':float(tail_motion[1]),
                'tail_joint_speed_rms_rad_s':float(tail_motion[2]),
                'clipped_control_steps':saturated,'max_target_clip_rad':max_clip,
                'clipped_joints':{model.actuator(i).name:int(n) for i,n in enumerate(clipped_joints) if n}}
        print('[LQR closed loop] '+json.dumps(report),flush=True)
        failures=rollout_motion_failures(tail_motion)
        if failures:
            raise RuntimeError(f'LQR {label} check: '+ '; '.join(failures)+
                               f'; clipped commands={saturated}; no artifact saved')
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_FILE)
    parser.add_argument('--seconds', type=float, default=5., help='Each of five CPU validation cases, >=3s.')
    parser.add_argument('--force', action='store_true', help='Explicitly replace an existing artifact.')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 3.:
        parser.error('--seconds must be finite and >=3')
    if args.output.exists() and not args.force:
        parser.error('Output exists; choose another path or explicitly use --force')
    import torch
    torch.set_num_threads(1)
    print('[LQR prepare] CPU only: equilibrium + native finite differences + five bounded flat-ground checks (including source pose + 15mm gap)', flush=True)
    model = make_model()
    q0, ctrl, eq = equilibrium(model)
    print('[LQR equilibrium] '+json.dumps(eq), flush=True)
    K, checks = gains(model, q0, ctrl)
    checks['rollouts'] = validate(model, q0, ctrl, K, args.seconds)
    metadata = {'schema':SCHEMA, 'source_sha256':source_digest(), 'validated':True,
                'gravity':9.81/6, 'physics_dt':float(model.opt.timestep), 'decimation':2,
                'equilibrium':eq, 'mujoco_version':mujoco.__version__,
                'model_options':{name:float(getattr(model.opt,name)) for name in
                    ('integrator','solver','cone','iterations','tolerance','impratio','disableflags','enableflags')},
                'scope':'local flat-ground double support; NOT validated on Site20', **checks}
    jids = model.actuator_trnid[:,0].astype(int)
    payload = dict(K=K, qpos0=q0, ctrl0=ctrl, keep=KEEP,
        joint_names=np.array([model.joint(i).name for i in range(1,model.njnt)]),
        actuator_joints=np.array([model.joint(i).name for i in jids]),
        kp=model.actuator_gainprm[:,0], kd=-model.actuator_biasprm[:,2],
        force_limit=model.actuator_forcerange[:,1], joint_range=model.jnt_range[1:],
        body_mass=model.body_mass, metadata=np.array(json.dumps(metadata)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=args.output.parent, suffix='.npz')
    os.close(fd)
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, args.output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f'[LQR saved] {args.output.resolve()} | radius={checks["spectral_radius"]:.8f}', flush=True)


if __name__ == '__main__':
    main()
