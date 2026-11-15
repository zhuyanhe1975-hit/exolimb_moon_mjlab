**最新入口与默认值（覆盖下方旧说明）**：

```bash
./run_sim2sim.sh --terrain moon --num-envs 16  # 一块16m月面，16套独立物理/策略/蒙皮
./run_sim2sim.sh --terrain moon                # 单人月面
./run_sim2sim.sh --terrain flat                # 平地对照
./run.sh stand-moon                           # MJLab月面回放，自动迁移默认开启
```

标准参数`--num-envs`，兼容`--num_envs`。用户最新要求：多人只共用**一块16m×16m月面和一个地面碰撞体**，不使用2km分散地图，不为每个人复制地面。`--patch`选择该块；在块内±5.5m的扩展安全池随机选择XY/yaw，完整鞋底覆盖检查、速度清零，出生根位置间距至少1.5m；复位也重新采样并避开其他人的当前位置。原生CollisionGroup排除人体之间的鞋底互碰，只保留鞋地面接触。默认每次启动用新随机种子，`--seed 42`可复现。多人默认总览，A切换总览/近景，逗号/句号切换实例，反斜杠开关跟随，鼠标自由转动/缩放；显式`--follow`启动近距跟随。无位置球/编号。相机复用`~/myWorks/zbot_exoarm_moon/.../tools/follow_camera.py`，通过`ZBOT_MOON_DIR`覆盖参考项目路径。

16人共享块CPU审计见`outputs/sim2sim_isaacsim/shared_patch16_static/multi_audit.json`：16 articulation、368 hinge、144独立skin mesh、33 collider（32鞋+1地面）；逐人体原生运动学/USD位置、出生间隙/间距、批量/逐人策略输出均校验。旧`multi16_static`是已废弃的多块布局审计，不作当前布局证据。旧多人bundle必须重新prepare，`shared_patch_v1`标记防止误读旧布局。已修复prepare包搜索路径；当前多人只导出共享块，不再导出2km场景。审计不启动Kit/窗口/PhysX步进，16人动力学仍待用户运行验证。

**控制离散化修正**：策略仍60Hz，PhysX/显式PD默认480Hz（每个源120Hz区间细分4次，`--physics-substeps 4`），每子步读回状态并重算原增益/原限幅力矩，读回核对armature及禁用的隐式drive。原120Hz显式PD冻结线性系统谱半径约2.76；480Hz检查小于1，仅是无接触、无饱和的近似分析，不是完整站稳证明。报告`outputs/sim2sim_isaacsim/control_audit/report.json`。用户自行运行的`run-20261115-101212-rWS1CH/report.json`记录平地6个10s回合，共60s/0失败，躯干线速度RMS约.00452m/s、角速度.01825rad/s；不是助手启动，也不能推广为16人随机月面已通过。

**迁移默认开启（用户最新要求）**：`run.sh stand-moon`缺失/不匹配快照时打印提示后使用当前地形，`--strict-terrain`恢复严格拒绝。训练入口仍保留显式迁移约定。sim2sim默认auto优先校验保存bank，否则自动使用当前月面；moon直接指定当前月面，current旧写法兼容；trained严格读取保存bank。多人用moon。迁移不等于复现旧训练地形，旧300代失配训练仍无效。

**MJLab刷新开销修正**：默认原生月面查看器（单人/多人）都用独立CPU共享显示模型。人体所在块保留10cm，未占用块内部使用1m显示LOD，全部10cm边界节点保留，整图/蒙皮/阴影保持；训练碰撞、惯量、观测/奖励不变。`--full-detail-view`恢复全部细网格。单人提交地形三角形3,636,736→479,680；256次实际bank原生射线误差最大5.04微米，5项已有共享显示回归通过。报告`outputs/stand_lunar_v15/view_lod_audit/report.json`。这是显示细节取舍及静态工作量减少，未实测新刷新率，不称无损或吞吐提升，重启已有窗口生效。

**Isaac Sim 闭环 sim2sim 推理入口**：`run_sim2sim.sh`。这与下方只渲染静态场景的 `run_isaacsim.sh` 分开。由PhysX CPU推进人体动力学，146维/23轴策略，无LQR、无训练、默认无扰动、带蒙皮。默认60秒/10秒回合，失败或超时复位；打印来源/地形/频率/映射并保存JSON/NPZ。以下为初版单人记录；涉及120Hz、严格迁移默认和单人限制的旧文字以上文最新设置为准。

用户首次运行暴露两个Foot缺少PhysxContactReportAPI，接触视图创建失败，最终`NoneType.filter_count`。现已在play/initialize_physics之前通过原生`RigidPrim.set_enabled_contact_tracking(True, threshold=0)`启用报告；物理初始化后预检两个刚体/过滤接触视图，失败不进入策略循环。异常完整Traceback在Kit关闭前直接打印并写入`.failure.json`，`SimulationApp.close(exit_code=1)`保留失败退出状态（Kit快速关闭会绕过Python后续异常打印）。此次仅修代码和CPU回归，尚未重新运行PhysX验证。

```bash
./run_sim2sim.sh --terrain moon       # 旧策略在当前detail3月面闭环测试
./run_sim2sim.sh --terrain flat                            # 同一策略先做平地对照
./run_sim2sim.sh --terrain flat --agent zero                # 零动作PD基线
./run_sim2sim.sh --terrain moon --headless --seconds 60 --report outputs/sim2sim_isaacsim/evaluation.json
./run_sim2sim.sh --prepare-only --terrain moon  # 仅CPU导出，不启动Kit/物理
```

可用`--checkpoint-file /绝对路径/model_N.pt`或`--experiment-name`指定来源；默认按现有月面同版本规则选择最新检查点，不跨任务兜底。`--task stand-plane`选择平地v14实验。`--patch`选择16m局部碰撞块，`--yaw`指定初始角度（度），`--no-skin`看原生几何。每次生成独立`outputs/sim2sim_isaacsim/run-*`目录。默认`--terrain trained`严格读取检查点旁`terrain/bank.npz`并核对SHA256；当前已有model_200缺少可验证快照，因此无参数会明确报错，应使用上面的显式迁移命令。缺失快照不等于旧策略已经学过detail3，旧失配地形300代仍不得当作有效月面训练。`--terrain flat`同样标明是平地对照。

桥接保留原生质量/COM/惯量、关节轴与限位、动作限幅、归一化统计和按`actuator_trnid`映射的PD/力矩限制，控制60Hz、PD480Hz、月重力1.635m/s²。目标引擎原生接触法向力按实际鞋底TOUCH体积汇总。MuJoCo只接收PhysX根/关节状态，调用`mj_kinematics/comPos/comVel/subtreeVel/sensorPos/sensorVel`构造原生质心、角动量、脚位姿与速度；不推进MuJoCo物理、不生成替代接触、不使用隐藏稳定控制器。观测张量直接导出生产`balance_mdp.features`并去除旧预测2维、追加躯干COM线速度，标准root重力投影在此模型中等于native root/site的逆旋转。

本机MJCF原生导入器会把同体多铰链合成D6，无法直接保留训练的转轴坐标约定，因此USD显式建立串联23铰链，12个辅助连杆共额外0.0012kg、各惯量1e-6kg·m²；原人体质量惯量保持。PhysX接触/关节限位/摩擦求解及显式PD积分与MuJoCo仍有差异，不称等价复现。脚地面仅三类collider（两鞋+地形），无其他人体碰撞；细地形三角面与高度观测一致。此入口只加载选中的局部训练块，不复制2km全图碰撞，也不是照片级最终渲染场景。

静态核对脚本`scripts/audit_sim2sim.py`：MJLab环境先生成32姿态golden，Isaac环境用`--usd`验证串联USD对原生运动学、MuJoCo3.11/3.12观测、跨环境策略输出与9块蒙皮。报告见`outputs/sim2sim_isaacsim/lunar_static_check/{native_audit,usd_audit}.json`。仅CPU静态检查，未启动Kit窗口、PhysX动力学或训练，不能据此声称已站稳；运行时另外核对PhysX实际body位姿与原生传感器模型一致，超差直接报错，失败写`.failure.json`。

**当前默认月面 v15（146维观测）**：`stand` / `stand-moon` / `stand-v15`，日志`wangyu_moon_stand_lunar_v15`（扰动变体加后缀）。平地v14保留`stand-plane` / `stand-v14`，日志`wangyu_moon_stand_flat_v14`；旧143/151维模型不能直接resume到146维任务。

**最新网格版 detail3**（覆盖下方旧版记录）：训练日志`wangyu_moon_stand_lunar_v15_groundsync1_detail3`，保留8192环境、146维、10s及全部奖励。5m源DEM以CPU三次样条作连续重建，原生碰撞仍为10cm采样的MuJoCo高度场；插值不代表新增实测精度。程序细节改为全局坐标场：10m规范单元保存确定性特征元数据，邻块带padding参与优先级拒绝，小坑跨单元/裁剪边界连续；全局起伏不再按裁剪中心重抽，删除4～5m方形淡出。大跨度稀疏查询分组处理，不创建整张2km×2km的10cm网格。

64块地形生成后均保留，安全出生点从各块内部选择；目前共9029个候选，最少的块也有20个，XY再抖动±2.5cm、yaw随机，不再因某个细节三角面较陡就整块淘汰。全鞋底保守覆盖检查继续生效。静态256姿态无穿透/不足14.5mm间隙，最小实测约17.8mm。

显示/Isaac导出改为**一体焊接网格**：背景三角化加入64块16m细网格的边界，逐边保证背景恰有一个相邻面；细块与背景复用边界顶点索引。消除独立hfield显示块的侧壁/底面及浮点裂隙，训练碰撞保持原生局部hfield。play现在64个hfield资源+1个显示mesh，不再额外生成64个显示hfield；Isaac NPZ相应导出1个mesh（约182万顶点），远处仍为粗LOD，训练区域保持相同几何。精确落在网格边/顶点的原生ray可能因浮点谓词漏判，审计检查两侧并独立验证焊接拓扑。512次两侧检查，细侧对碰撞高度最大误差约0.009mm；跨2mm接缝高度变化最大约0.52mm。报告`outputs/terrain_generation_audit/detail3_verification.json`。

本轮完成全局场/拼接/出生点/原生刚体和快照等CPU静态检查及Isaac网格导出，未启动训练、GPU动态或Isaac窗口。旧模型需显式`--terrain-transfer`才能试用新地形，不算原地形复现；新结果使用detail3独立日志和指纹。可由用户运行`./run_isaacsim.sh`预览当前导出，或`./run.sh stand-moon --agent zero --num_envs 16`检查基本显示。多尺度坑形、真实岩块和远景高分辨率动态LOD仍未实现。

**Isaac Sim 场景预览入口**（不训练、不加载策略）：

```bash
./run_isaacsim.sh                               # 交互RTX窗口，低机位看月面
./run_isaacsim.sh --focus astronaut              # 查看静态望宇USD
./run_isaacsim.sh --quality path                 # 路径追踪，静止后累积画质
./run_isaacsim.sh --quality path --frames 160 --warmup 120 --snapshot outputs/isaacsim_preview/scene.png
./run_isaacsim.sh --dry-run                      # 检查运行环境和缓存资源，不启动Kit
```

参考`~/myWorks/zbot_exoarm_moon`启动方式，通过`uv run --project ~/nvidia/isaac-lab-3.0.0 --no-sync --offline`复用已有Isaac环境，隔离MJLab的Python 3.13。先由MJLab环境在CPU上把实际原生地形显示面（65个mesh/hfield顶面）导出至`outputs/isaacsim_preview/terrain.npz`，再由Isaac Sim Python 3.12创建USD网格、OmniPBR月壤材质与太阳光。坐标为当前地形块的局部米制Z-up，元数据保留源地图偏移和地形指纹。纹理使用参考项目已有LunarRegolith4k(diff/rough/normal/AO)，材质和1750/.53太阳参数参考OmniLRS LargeScale；不是启动整个OmniLRS应用，也不复制其地形替换训练地形。

默认引用参考项目`outputs/wangyu_new_preview/wangyu_rigged.usdc`静态望宇，仅用原始USD姿态预览，尚未接入MJLab关节/刚体轨迹；`--no-astronaut`只看月面。无SimulationContext、不播放timeline，app.update仅用于Kit渲染。`--export-usd path.usda`可保存当前场景（保留对原资源的引用）。可用`ISAACLAB_DIR`、`MJLAB_PYTHON`、`ZBOT_MOON_DIR`覆盖本机路径。脚本help/dry-run、两个Python版本的语法、65面CPU导出已通过，未启动Isaac GUI或声称画面已验收。

**当前物理修复版 detail2**：新训练日志`wangyu_moon_stand_lunar_v15_groundsync1_detail2`（扰动变体加后缀），146维/8192环境/10s及奖励不变。出生检查改用原生鞋底rest变换生成≤2.5cm采样网格，并按2m认证区域内真实三角面的最大坡度×覆盖半径补足样点之间的高度余量；越出认证区或非水平rest鞋底直接报错。保留10cm地形网格，坑缘sigma至少1.5格(FWHM≥3.53格)，凸起短径至少6格，避免未评估就加密GPU地形。CPU静态256出生姿态最小实际竖直间隙约18.14mm，均无穿透、无低于14.5mm；结果见`outputs/terrain_generation_audit/detail2_verification.json`，并非GPU动态验证。

新训练快照增加实际bank、配置/代码/版本指纹及archive SHA256；回放和月面续训加载前强制核对，缺失/不匹配/损坏会报错。旧策略在当前地形上试用须显式`--terrain-transfer`，日志明确标记迁移，不计为原地形复现；`--resume-from`平地v14是明确的平地→月面迁移入口。查看新地形无需策略可用`./run.sh stand-moon --agent zero --num_envs 16`。本轮未启动训练或动态回放。

**渲染分工（最新用户约定）**：MJLab只负责原生物理、训练、策略执行与基本调试显示，不继续追求照片级渲染；本轮试加的月壤材质/纹理/灯光已撤回。照片级目标参考OmniLRS的`environment=largescale`，放到训练完成后的Isaac Sim回放阶段。推荐接口为MuJoCo原生轨迹→Isaac Sim姿态还原/渲染，记录时间、reset标记、源地图全局body位姿、23轴状态及地形快照，避免跨reset插值。该导出/Isaac导入链路尚未实现；若要在PhysX中闭环执行策略，另需sim2sim动力学适配，不能宣称直接加载权重即等价执行。

**2026-11-15地形生成审计**：对照本地OmniLRS并运行CPU静态原生碰撞查询，报告`outputs/terrain_generation_audit/report.json`，复现脚本`scripts/audit_lunar_generation.py`（不调用mj_step、不训练）。确认：50脚点的15mm出生间隙检查不足以保证全鞋底间隙，256静态姿态中12个低于14.5mm、最小约11.3mm，但均无初始穿透；最窄坑缘FWHM仅1.13个地形网格间隔；212块候选中148块被细节坡度整块过滤；细节仅覆盖整图0.16%；crop独立种子不能作为可任意重叠查询的全局地形场；内部方形淡出带对角线上存在导数折线；回放没有加载或核对训练保存的bank，改参数/算法后可能悄然换地形。单一坑形/半径分布、平滑凸起以及保留的5m三角面折痕也限制真实感。当前显示/碰撞CPU射线一致性3项仍通过；降低20mm后的CPU接触最多43，未超过现有128预算，不等于GPU动态接触已验证。本轮只新增审计，没有修改默认地形或训练配置。

**多人共享月面回放**：`./run.sh stand-moon --num_envs 16`在同一窗口显示一张Site20及16个人体，不分屏、不重复绘制地形。CPU原生显示模型为每个人保留独立的skin/bone，复用3份纹理；按源地图中心及海拔将GPU局部根位姿平移回实际月面位置，朝向与关节姿态原样保留。原生查看器不再用`mjv_addGeoms`把其他世界的同一套skin叠上来。按 **A** 切换全图总览，按 **逗号/句号** 切换近距离跟随，滚轮缩放。人物分散在真实2km地图内，近景通常只看到当前跟随者。回放<=64环境时初始和局部复位选择不重复的地形中心，仍随机XY/yaw；训练8192独立样本与奖励/观测不变。鼠标拖拽按显示body映射到对应仿真环境，力/力矩显示保留。

`outputs/stand_lunar_v15/shared_view/overview.png`、`tracked.png`为16人静态原生渲染，数据见同目录metadata.json；`scripts/preview_shared_moon.py`只渲染不推进物理。CPU原生单测核对骨骼命名归属、根/刚体坐标平移、共享地面射线和鼠标广义力映射。此次未启动动态回放或训练。

按用户要求，总览和近景均不绘制位置球或人体编号文字。A总览、逗号/句号切换跟随和原有扰动力/力矩可视化保留。

总览后的蓝色遮挡已确认来自**A键同时触发MuJoCo原生Auto Connect**，先前仅删除位置标记没有解决根因。共享查看器在初始化、A操作及每帧关闭`mjVIS_AUTOCONNECT`；显示模型的原生connect颜色设透明，避免UI线程先处理A时短暂闪现。`outputs/stand_lunar_v15/autoconnect_repro/{before,after}.png`为同一16人静态姿态、强制打开该原生标志的修复前后图，audit.json记录条件。原生场景与A状态回归通过，无训练或物理步进。

**最新地形细节版 `native_groundsync1_detail1`**：参考本地`~/Github/OmniLRS`的分层DEM、小坑泊松采样/间距拒绝、变形径向坑形和多尺度起伏方法，独立用NumPy预生成，不导入OmniLRS/Isaac Sim、不复制其源码或资源。保留原Site20 2km地形，在64个地形块增加程序生成的亚5m细节，不能将它们当作Site20实测数据或经过标定的月壤模型。

- `assets/site20/detail_config.json`设置固定种子、坑密度/半径/深度、坑缘高度、起伏与凸起参数；默认10cm采样，坑半径0.3～1m、深度为半径的7.5%，坑缘为深度的30%，嵌入式凸起高1～3.5cm。凸起属于高度场，不是可移动石块。
- 固定源地图坐标决定种子，reset只选择预生成地形；训练与高度观测读取同一批原生hfield。中央候选出生区按加细后的三角面坡度≤15°筛选，再检查旋转脚底50点间隙；8192默认环境、146维、10s及奖励不变。
- 为避免原粗地形把负向小坑填平，完整DEM显示采用保留原5m三角面的原生mesh，在64处10m方形区域用细高度场替换；细节在4～5m边带平滑降至0。没有整个2km的10cm网格，远处原始DEM不变。只有当前局部块参与碰撞，64个细节显示块及背景均不碰撞。
- 新训练日志隔离为`wangyu_moon_stand_lunar_v15_groundsync1_detail1`，自定义扰动继续追加幅度后缀。每个训练run的`terrain/metadata.json`、`bank.npz`保存参数SHA256、源信息和实际高度数据，方便追溯；不是在训练每步重新生成细节。
- 静态测试通过：种子复现、无细节外区域保留源高度、原生射线检查细节显示/碰撞及背景接缝、256假环境批量reset。原生静态渲染已检查，生成高差约−7.7～+5.6cm。**新细节版未进行GPU动力学或训练验证**，不能沿用之前无细节版GPU检查结论。

预览：`outputs/stand_lunar_v15/detail/standing.png`、`surface.png`；元数据`metadata.json`。显式运行静态预览（不推进物理）：

```bash
~/dev-envs/robotics-mjlab-py313/bin/python scripts/preview_lunar_detail.py
./run.sh stand-moon
```

源方法对应OmniLRS的`src/terrain_management/large_scale_terrain/crater_distribution.py`、`crater_generation.py`和`src/terrain_management/terrain_generation.py`。旧300代碰撞失配结果仍无效，后续应从平地model_500重新迁移，不能把旧模型当成已在细节地形上训练。

以下为细节加入之前的局部地形及修复记录（最新实现见上文）：月面使用Site20真实DEM的64块16×16m局部原生MuJoCo高度场，各环境独立随机地形块、XY和yaw。当时的0.25m碰撞采样仅细分原5m数据，没有新增地形细节。局部坐标避免旧2km高度场的巨大碰撞棱柱；逐脚底采样检查出生间隙，并保留1m活动范围及地形边界保护。当时8192环境出生检查通过，不代表当前细节版已经动态验证。

回放显示已恢复**完整原始2km×2km Site20 DEM**，不再只显示16m截块；全图随选中的源DEM中心和海拔平移到同一局部坐标系，没有缩放或夸张高度。全图只显示（contype/conaffinity=0），脚下仍由匹配的16m局部高度场碰撞，局部碰撞几何透明避免重叠。它不表示全图都能行走，原地实验仍限制距出生点1m。静态原生射线在4个地形中心、各3点核对全图显示面与局部碰撞高度，误差小于40微米；未在此改动后进行动态回放或新训练。

**地面同步修正 native_groundsync1**：初版v15把地面挂在world下；本机MJWarp对这类静态geom仅初始化一次`geom_xpos`，reset改`geom_pos`不会刷新GPU碰撞高度，导致人物掉落。现用静止的原生mocap父body，由原生运动学刷新地面位置，无地面动力学自由度。全部64地形变体两轮GPU位置/接触检查通过；`scripts/audit_local_lunar_runtime.py`可显式执行这个有界回归。修复后原model_799的16环境×5秒无扰动测试有1次失败复位，不能称已稳定。报告`outputs/stand_lunar_v15/groundsync_evaluation.json`。

下述300代的碰撞高度有误，**不构成有效月面训练结果**，原报告和运行来源已标记`invalid_terrain_training`，保留文件供追溯。当前run仍可加载该模型检查修复后的物理，并打印警告；之后训练应从原平地model_500重新迁移。新训练日志隔离到`wangyu_moon_stand_lunar_v15_groundsync1`（扰动变体追加后缀），run优先选择新日志中已保存的匹配模型。本次只修正和有界验证，没有重新训练。

2026-11-14按用户要求，从平地`2026-11-14_21-56-39/model_500.pt`恢复策略、归一化与优化器，在月面续训300次更新（编号500..799），保存于`logs/rsl_rl/wangyu_moon_stand_lunar_v15_force200_torque50/2026-11-14_22-18-13/model_799.pt`。沿用8192环境、10s回合、146维观测、200N/50Nm扰动及原奖励。首/末50代平均奖励−1.55→−1.05，但躯干速度仍约0.47m/s、跌倒频繁，**尚未站稳**。这是训练统计，未另做确定性评估。报告及完整日志见`outputs/stand_lunar_v15/resume300.json`、`resume300.log`；运行目录另存续训来源SHA256与实际地形数据。

```bash
./run.sh stand-moon                       # 最新月面结果，默认无扰动
./run.sh stand-moon --pushes --push-force 200 --push-torque 50
./run.sh stand-plane                      # 保留的平地结果
```

用户要求把躯干运动与root重力方位直接提供给策略：

| 观测（零基列） | 含义/坐标系 | 缩放 |
|---|---|---|
| 46:49 | root坐标系重力投影，复用G1同款`mjlab.envs.mdp.observations.projected_gravity`，替换原有等价3列，不重复追加 | 单位方向向量，模长1 |
| 55:58 | 已有Torso三轴角速度，原生FRAMEANGVEL转Torso局部系，保留 | ×0.5 |
| 143:146 | 新增Torso刚体质心线速度，原生FRAMELINVEL/BODY Torso世界系量，通过标准quat_apply_inverse转Torso局部系 | ÷0.3m/s |

root重力投影为R_world_root转置乘[0,0,-1]，与月面1.635m/s²大小分开；能感知roll/pitch及倾斜方向，不能确定绕重力的绝对yaw。Torso角速度此前已在143维中，因此只新增3维线速度：143→146，而非重复添加角速度/重力变成152维。actor/critic一致；观测中的方向信息不等于恢复倾角奖励。稳定裕度/LIPM预测仍全部不进入观测/奖励。全部奖励、10s、8192、物理与扰动开关均保持v13。

CPU原生kinematics/comVel/sensorPos/sensorVel静态对照验证局部速度方向、roll/pitch/yaw重力投影与归一化，以及原始角速度所在列；不做动力学积分/训练。报告和异常现场`outputs/stand_flat_v14`。

```bash
./train.sh stand --pushes --push-force 200 --push-torque 50 --agent.max-iterations 2000
./run.sh stand
```

**当前默认平地实验为 v13：稳定裕度不再进入观测或奖励。** `stand` / `stand-plane` / `stand-v13` 使用 `Wangyu-Moon-Stand-Flat-v13`，日志基名 `wangyu_moon_stand_flat_v13`（扰动变体加后缀）。

输入151→143维：移除新增6维支撑几何扩展（静态/捕获裕度、捕获偏移XY、接触槽占比、面积有效标记），同时移除旧145维中的2维LIPM预测偏移。保留关节/速度/重力方向/原生质心状态/角动量/脚部位置速度/接触载荷/动作等基础状态，actor和critic一致。capture_risk奖励项彻底删除，相关凸包/捕获裕度指标不再在默认训练与评估路径计算；airborne复用同一原生接触有效性掩码直接判断，不经过凸包。历史balance bundle仍可计算其内部诊断，但LIPM预测列不提供给新策略。

平地物理、8192环境、10秒、躯干线速度-4、角速度-.2、H-.01、joint_pose-.25及其他权重保持。输入维数变化，旧151维模型不能直接resume到v13，需要新训。旧v12入口保留`stand-v12`，可配`--experiment-name`选此前151维结果。

```bash
./train.sh stand --pushes --push-force 200 --push-torque 50 --agent.max-iterations 2000
./run.sh stand
```

仅做CPU功能/配置/命令入口检查，未训练或仿真。v13报告/异常现场在`outputs/stand_flat_v13`。

用户最新有效设置：`capture_risk=0`（位于support_task.REWARD_WEIGHTS，平地继承），关闭稳定裕度奖励。用户反馈效果更好、摇摆减少；该反馈尚未由助手独立复测。151维观测及接触/裕度诊断仍保留，躯干线速度/角速度、质心角动量等其他项不变。后续默认保持关闭，不自动恢复此奖励。

用户最新要求增加躯干角速度惩罚：当前平地v12增加`torso_angular_velocity=-0.2`，计算现有原生FRAMEANGVEL传感器`balance_torso_omega`三轴平方和（roll/pitch/yaw都计入），单位rad²/s²。不是倾角惩罚；gravity_tilt仍0，允许静态倾斜。保留torso_velocity=-4、H=-.01、joint_pose=-.25、10s、151维和全部物理。新日志基名`wangyu_moon_stand_support_flat_v12_contactsync1_quiet1_torso2`，新增torso_angular_speed_rad_s指标及motion-trace中的Torso角速度。已有v12可显式resume，新旧结果隔离。静态native角速度与mj_jacBodyCom旋转雅可比*qvel一致，另验证三轴都受罚；未训练/仿真或宣称动态效果。

用户最新要求：平地v12将骨盆线速度惩罚替换为`torso_velocity=-4.0`，删除pelvis_velocity奖励项。新增原生FRAMELINVEL/BODY Torso传感器`balance_torso_com_vel`，读取Torso刚体惯性质心的世界系线速度，计算全三轴平方和；不是腰部原点速度，也不是上身子树平均速度。H=-.01、joint_pose=-.25、10s、其他权重/物理/151维观测不变。新增`torso_speed_m_s`指标及motion-trace中的torso速度。骨盆速度仍可作为诊断输出，已不参与此项惩罚。新日志基名`wangyu_moon_stand_support_flat_v12_contactsync1_quiet1_torso1`，支持已有v12显式resume隔离保存。静态原生运动学测试证实：仅腰转动、骨盆速度为0时，Torso COM速度非0，且与原生mj_jacBodyCom*qvel一致；无forward动力学/步进/训练。

用户最新要求提高运动惩罚：当前平地v12的pelvis_velocity=-4.0（原-.5，提高8倍），centroidal_angular_momentum=-.01（原-.002，提高5倍）；joint_pose=-.25等其他权重、10秒回合、物理与151维观测均不变。实际配置统一取flat_support_task.REWARD_WEIGHTS，启动打印一致。新日志基名`wangyu_moon_stand_support_flat_v12_contactsync1_quiet1`，自定义扰动继续加幅度后缀；可用--resume-from从现有v12检查点续训，不覆盖旧实验。改权重只影响之后学习，不会改变已经固定的策略输出。

用户已授权的一次有界诊断（非训练）：model_300，无扰动确定性64环境×30秒，零失败复位；出生2秒后骨盆线速RMS .02258m/s，角速 .11801rad/s，H RMS1.46054kgm²/s，关节速RMS .09553rad/s。原权重的实际惩罚率分别为骨盆-.0002515、H-.0042703；确实生效，但缓动代价小，且H总量不等于每个关节都静止。零动作PD对照64×30秒全部出现失稳、共384次失败复位；不能将固定源关节角当作已验证平衡控制。另做同权重10秒回合的30秒回放（超时/失败分开统计），原始报告和native轨迹在`outputs/stand_support_flat_v12/motion_check/`。该诊断没有验证提高权重后的学习效果，也没有启动新训练。

回放新增`--evaluate --motion-trace`导出native运动和已计算的RewardManager惩罚率（不重算奖励），`--episode-seconds 10`保留10秒超时复位；缺省仍长窗不超时，便于看收敛后的残余运动。注意报告passed仅指存活，不代表静止。

用户最新设置：当前平地v12训练/普通回放的回合上限为10秒（60Hz控制，共600步）。独立`--evaluate --seconds 60`仍按指定60秒评估。

**接触时序修复 contactsync1 已生效**：`support_mdp.phase_contact_data`在post-forward观测阶段通过原生公共`ContactSensor.update(dt=0)`失效缓存，再读取同阶段native sensordata；奖励仍使用pre-forward状态。没有额外forward、物理时间推进、共享依赖修改或逐环境循环。该传感器必须关闭air-time/history（当前配置均关闭），否则明确报错以免多写历史帧。actor/critic共享同阶段缓存，仅刷新一次；partial reset按reset generation失效。

原有151维观测/23动作/奖励权重/物理均保留，已有平地v12检查点可显式加载或用`--resume-from`续训。修复后平地新日志基名为`wangyu_moon_stand_support_flat_v12_contactsync1`，扰动变体加后缀；v10/v11新训练目录也加contactsync1，不覆盖旧结果。默认run只查修复后的目录，尚无新结果时可显式`--checkpoint-file`或`--experiment-name`选已有v12结果。示例（由用户执行）：

```bash
./train.sh stand --pushes --push-force 200 --push-torque 50 \
  --resume-from logs/rsl_rl/wangyu_moon_stand_support_flat_v12_force200_torque50/2026-11-14_20-39-20/model_200.pt \
  --agent.max-iterations 100
```

增加5项真实ContactSensor+CPU缓冲回归测试：起跳、质心/接触同步平移、下一物理步奖励、局部复位、禁止stateful历史刷新；另有原6项几何/配置测试。原生静态审计调用生产刷新函数验证修复后结果，未执行mj_step/GPU/策略/训练。裕度仍是LIPM投影近似，此次不改变载荷门限、height定义或奖励公式，不把它升级宣称为严格ZMP判据。

稳定裕度独立审计：`<shared-python> scripts/audit_support_margin.py`，输出`outputs/support_margin_audit/report.json`和`static_support.png`。3000组随机点集对SciPy凸包/线段距离：float32最大误差约8e-8m；静态原生MuJoCo快照（mj_forward但无mj_step/时间推进）核对接触传感器、原生subtree COM/速度与独立公式。

**此前审计发现观测接触缓存时序问题，现已按下述contactsync1修复**：`support_mdp.data`虽分reward/observation阶段缓存，但`ContactSensor.data`内部torch.cat结果另有缓存。Env在reward后sim.forward/sense刷新底层传感器；无reset时没有scene.update去失效该缓存，因而新COM可能搭配旧接触点。静态对照中，倾身快照的错误裕度相差约2.43cm；离地快照仍可能拿旧接触报+13.52cm，而真实已无接触。部分reset会使整个ContactSensor缓存失效，因此问题不一定每步出现；不能将静态对照误差当成实训统计。原始审计本身不修改训练；后续contactsync1修复了观测缓存刷新，权重和物理不变。

还需区分几何与物理含义：目前只是LIPM捕获点到有载接触点凸包的投影距离，不是ZMP/一般可恢复性判据；依赖恒质心高度/忽略角动量变化和持续外力矩。当前force.norm>.5N筛选合力而非法向载荷，选取的是即时承载子集，不等于所有可用支撑点。无接触-.2m是哨兵值，不是实际几何距离。平地以接触中点平均z而非精确plane z计算h会有小偏差（0.12m/s静态快照约0.12mm裕度差）。这些发现优先于继续调奖励权重。

**当前默认切回原生平地站立 v12**（`stand` / `stand-plane` / `stand-v12`）：无LQR、原单块鞋底、无限plane，完全不加载DEM。默认8192环境、23轴、月面重力1.635m/s²、30秒回合。保留151维观测和当前支撑目标/全部权重（joint_pose=-.25，质心角动量=-.002，gravity_tilt=0），保留数值异常保护及无扰动静止诊断。固定源姿态/零速度，无随机地形位置或yaw；原生静态mj_kinematics+脚底到plane的mj_geomDistance确定出生根高度，最小脚底间隙15mm。不施加LQR或额外支撑。

```bash
./train.sh stand --no-pushes --agent.max-iterations 2000  # 先验证无扰动站立
./run.sh stand                                         # 默认无扰动，自动找同版本最新结果
./run.sh stand --evaluate --num-envs 256 --seconds 60
# 平地大扰动实验：
./train.sh stand --pushes --push-force 200 --push-torque 50 --agent.max-iterations 2000
./run.sh stand --pushes --push-force 200 --push-torque 50
```

不指定扰动参数时，训练仍沿用±30N/零额外力矩，run默认关闭；上面的无扰动训练显式`--no-pushes`。日志隔离到`wangyu_moon_stand_support_flat_v12`，无扰动/自定义幅度分别加`_no_push`/幅度后缀；需新训或只续训v12，不加载旧地形策略作为默认结果。原DEM v10用`stand-terrain`/`stand-v10`，四分块高度场v11仍是`stand-contact`/`stand-v11`。v12报告及异常现场在`outputs/stand_support_flat_v12`。已静态编译真实train/skin场景，确认无hfield、接触传感器对象存在、质量/关节数正确、原生脚底距离和重力正确；未运行训练/仿真，不宣称已解决动力学发散或晃动。

v11启动保护报错排查（用户实际现场，非助手仿真）：`outputs/stand_support_contact_v11/anomalies/anomaly_20261114_202605_766576_3371003.pt`显示第12控制步env956、原生施加外力为0；一个接触深度约-9m、位置离脚底数米、力约236kN，随后关节角速度达到298rad/s。保护停止是预期行为，不能禁用保护继续PPO。v10也有独立异常现场，不能把所有问题都归因于四分块。

`stand-contact`新增CCD容差对照profile：通过scene根MjSpec.option设置原生绝对ccd_tolerance=1e-3m（原1e-6），启动打印；未改时间步/PD/奖励。~1km float32坐标的分辨率约0.1mm，1um小于该分辨率；调容差是有依据的待验证对照，**尚未确认消除虚假接触**。本机MJLab MujocoCfg不暴露ccd_tolerance，所以不能简单添加一个无效cfg属性；已静态验证根spec设置经cfg.sim.mujoco.apply后保留。新日志隔离到`wangyu_moon_stand_support_contact_v11_ccd1e3`（自定义力矩另加幅度后缀），原模型/异常现场保留。命令`./train.sh stand-contact --agent.max-iterations 2000`；`stand`仍是旧v10，不会启用此接触profile。形状探测阶段在首次地形reset之前仅返回151维零张量供MJLab检查维数，不作为策略观测；真实reset/运行的异常检查仍严格。

**新增独立鞋底接触对照 v11（`stand-contact` / `stand-v11`）**：

```bash
./train.sh stand-contact --agent.max-iterations 2000
./run.sh stand-contact
./run.sh stand-contact --evaluate --num-envs 256 --seconds 60
# 保留原来的大扰动配置：
./train.sh stand-contact --pushes --push-force 200 --push-torque 50 --agent.max-iterations 2000
```

v11将每只原生box鞋底等分为4个无缝box碰撞块，总外形/厚度不变；用MuJoCo编译出的原始foot质量、COM、主惯量和惯性轴显式固定惯性，避免分块改变动力学质量属性。摩擦、接触参数、PD、关节范围、151维观测、现有奖励全部保持v10设置（joint_pose=-.25，质心角动量=-.002）。传感器按Foot body聚合所有碰撞块，每脚保留4个真实有载接触槽，不补假接触或额外支撑。默认8192环境、30秒回合、月面重力，只有鞋底与地面碰撞。静态编译验证质量/惯量/关节与执行器完全一致，鞋底4块构成原box精确分区，真实robot/蒙皮挂载可编译。

原因线索：本机Warp的heightfield CCD对每个地形三角棱柱先生成一个候选点，再从不同棱柱的候选点中选取多个接触；5m DEM网格比鞋底大得多，整只box鞋底可能只有一个候选点。传感器slots=4不会制造接触；四分块使多个独立原生碰撞对有机会产生分布在鞋底上的接触。此为静态源码分析与待验证方案，不宣称已消除抖动，也不保证所有坡面全掌接地。

v11物理碰撞模型变化，日志隔离至`wangyu_moon_stand_support_contact_v11`，自定义幅度加后缀；需新训，不直接resume已有v10。原stand仍为v10，便于同条件对照。v11异常现场保存于`outputs/stand_support_contact_v11/anomalies`。尚未运行仿真/训练或吞吐测试；分块增加碰撞对，性能影响待测。

v10新增异常保护与无扰动静止诊断（不改151维观测含义、物理或奖励权重，旧策略可加载）：

- 复用本机NanGuard原生状态非有限值检测；补充原生关节限位超出1rad、角速度>200rad/s、根线速度>50m/s、无效根四元数、原生加速度>1e8的粗异常检查。这些是数值异常阈值，不是站立目标或常规摔倒判据。
- 检查位于首个终止项（奖励之前）与post-forward观测入口；观测非有限或绝对值>1e5也停止。触发时保存未经裁剪的qpos/qvel/qacc/warmstart/ctrl/sensors/xfrc、动作、先前输入观测、接触、原因和原生MJB模型到`outputs/stand_support_v10/anomalies/`（可用WANGYU_DIAGNOSTIC_DIR改位置），随后抛错停止，异常转移不会进入PPO process_env_step/update。不裁剪异常奖励继续学习，不将异常隐藏成正常摔倒复位。控制步边界检查并不保证抓到每个物理子步内的瞬时异常；未定位具体积分/接触根因。诊断会产生GPU同步，未测吞吐影响；不启用全8192环境的100帧native NanGuard滚动副本。
- 增加joint_speed_rms_rad_s、root_angular_speed_rad_s、loaded_contact_slots、settled_unforced_fraction和moving_without_wrench_fraction。后两项排除出生/原生外力或额外广义外力结束后2秒；moving判据为根线速>.03m/s、根角速>.1rad/s或关节RMS速度>.1rad/s任一成立。TensorBoard moving项是全采样时间中的占比，可与同窗口settled项相除得到条件占比；不是新的奖励。
- `./run.sh stand --evaluate --num-envs 256 --seconds 60` 默认无扰动；报告`settled_unforced`直接给有效样本数、条件运动比例和三种RMS速度。无有效样本时输出null，不伪报0。新诊断不会自动修复已经学到的摆动策略，需结合原始异常现场继续定位。
- CPU假状态/配置与静态模型保存读取已验证，不推进物理、不执行策略/训练。

run检查点选择更新：v8/v9/v10各自在同任务版本的默认及自定义扰动目录中选择最新非smoke运行的最大模型编号。显式指定`--push-force/--push-torque`且有匹配的已保存实验时优先选择匹配目录，否则选择该版本最新保存运行；启动打印选择理由。`--experiment-name`固定指定目录，`--checkpoint-file`固定指定模型。不跨任务版本回退。

用户最新约定：v10保留原生整个人体质心角动量平方惩罚 `centroidal_angular_momentum=-0.002`，同时保留 `joint_pose=-0.25`；持续倾角惩罚仍为0。

用户最新约定：v10保留初始关节角偏离惩罚 `joint_pose=-0.25`，由 `support_task.REWARD_WEIGHTS` 统一设置；倾角与角动量持续惩罚仍为0。

当前主线 **v10 支撑恢复实验**（`stand` / `stand-v10`），旧v9用`stand-v9`，旧v8用`stand-v8`。

```bash
./train.sh stand --agent.max-iterations 2000
./train.sh stand --pushes --push-force 200 --push-torque 50 --agent.max-iterations 2000
./run.sh stand
# 自定义大扰动实验的结果：
./run.sh stand --experiment-name wangyu_moon_stand_support_v10_force200_torque50 --pushes --push-force 200 --push-torque 50
```

v10新增6维观测（总151），须新训练，不能直接续训v8/v9。8192环境，30秒回合，地形/出生安全/物理/PD保持原设置；run仍默认关闭扰动。权重以`src/support_task.py:REWARD_WEIGHTS`单一字典为准，打印与实际一致。

v10目标：gravity_tilt=0、joint_pose=-0.25、centroidal_angular_momentum=-0.002；pelvis_velocity=-0.5、alive=+0.2、action_change=-0.002、joint_torques=-2e-6。新capture_risk=-0.15，airborne=-0.2。capture_risk为`min(4, [max(0,0.02-margin)/0.1]^2)`，鼓励2cm内部缓冲，允许躯干倾斜和临时角动量恢复。倾角只用于观测和60°失败保护，不持续要求身体竖直。

原生MJLab ContactSensor每脚保留4个最大载荷接触槽（共8），使用真实接触位置、found、接触力模长>0.5N、间隙<=2mm筛选；不补鞋底角点。求水平投影接触凸包，正裕度在区域内，负值为区域外距离；点/线支撑没有正面积和正内部裕度。截取接触点是完整接触集合的子集，不扩大区域。无接触显式无效并额外惩罚，不伪造稳定面积。

质心/速度读取原生subtree传感器，捕获点近似为`com_xy + sqrt(h/g)*com_velocity_xy`；h相对有载接触点平均高度。这是LIPM投影近似，**不是一般非共面摩擦/力矩约束下的可行域或严格ZMP判据**，不把外加力矩代入一个假定成立的ZMP公式。接触几何和两种裕度、捕获偏移、接触槽占比、支撑面积有效性进入观测/日志；偏移转换到出生动作一致的当前yaw局部系。退化接触情况下按真实点/线计算，需监测`support_has_area`。基础物理量不手工重算。固定张量几何支持项目已有CUDA Graph缓存，无每环境CPU凸包/LP循环；未运行GPU吞吐或训练验证。

v10评估将“有限时长无失败复位”与姿态/速度统计分开，不再用3°倾角或5cm固定站位作为通过条件；输出捕获裕度近似、区域外时间比例、无载接触比例和非退化区域比例。30秒训练回合仍不能替代独立60秒评估。

训练扰动参数同样支持 `./train.sh stand --pushes --push-force 200 --push-torque 50`。自定义扰动隔离到 `wangyu_moon_stand_push_v9_force200_torque50`。续训加 `--resume-from /absolute/path/model_2098.pt --agent.max-iterations 100`（额外训练100代，恢复优化器与模型状态）。不加自定义参数仍沿用原训练±30N/零力矩。新分布的检查点回放可用 `--checkpoint-file` 显式指定。

native力矩显示：紫色箭头表示实际世界系力矩轴，圆弧箭头表示右手定则的旋转方向，标注合力矩大小Nm；长度随大小变化并作显示限幅。读取选中环境的原生xfrc_applied后三维，不修改施力或策略。无力矩时消失；使用 `./run.sh stand --pushes --push-torque 10`。

回放/评估随机外力与力矩：`./run.sh stand --pushes --push-force 60 --push-torque 10`。每次脉冲独立均匀采样世界系Fx/Fy各±60N（Fz=0）及Tx/Ty/Tz各±10Nm，同次脉冲内保持采样值，持续0.10–0.20s，冷却2–4s，各环境独立计时。`--push-torque 0`关闭附加力矩；`--push-force 0`可仅测试力矩。默认run无扰动，训练配置保持原设置。

回放/评估扰动力幅度：`./run.sh stand --pushes --push-force 60`。`--push-force` 为世界系Fx、Fy各自的绝对上限（N），启用扰动时默认60，即原30的两倍，最大合力约84.9N。可设任意非负有限值；启动信息与评估报告记录实际范围。此覆盖仅影响回放/评估，训练仍沿用原±30N，不改变现有实验分布。run默认仍不施加扰动。

原生查看器默认开启MuJoCo外力箭头显示，读取实际 `xfrc_applied`，仅施力期间出现。使用 `./run.sh stand --pushes` 查看随机骨盆扰动力；默认无扰动时不会出现随机外力箭头。此功能不改变物理或策略，当前针对native查看器。

当前默认为 **v9**：仅将 `joint_pose` 初始关节角偏离奖励权重设为0，其他物理、观测、奖励及训练扰动保持v8设置，8192环境。`./train.sh stand` / `./run.sh stand` 使用v9；旧实验用 `stand-v8`。回放/评估仍默认关闭扰动，`--pushes` 开启。v9日志与报告分别位于 `logs/rsl_rl/wangyu_moon_stand_push_v9` 和 `outputs/stand_push_v9`。

回放扰动开关（默认关闭，只影响本次回放/评估，训练配置与检查点不变）：
```bash
./run.sh stand --no-pushes                  # 关闭随机外力（默认）
./run.sh stand --pushes                     # 开启随机外力
./run.sh stand --evaluate --num-envs 256 --seconds 60 --no-pushes
```
启动时打印实际扰动状态。无扰动评估默认保存 `outputs/stand_push_v8/evaluation_no_push.json`，与有扰动结果分开。

当前主线：无LQR月面站立＋骨盆随机水平力扰动，任务 `Wangyu-Moon-Stand-Push-v8`。
`./train.sh` / `./train.sh stand` 默认进入v8；23轴、8192环境、月面重力1.635m/s²，原地形/出生安全检查/观测/奖励保持。

```bash
./train.sh stand --agent.max-iterations 2000
./run.sh stand                            # 最新v8结果，回放也有随机外力
./train.sh stand-v4                       # 原无扰动v4对照
./run.sh stand-v4                         # 原无扰动v4结果
```

扰动施加在Pelvis刚体质心，世界系Fx、Fy各独立均匀采样[-30,30]N（合力最大约42.4N）；Fz=0、额外力矩=0。
每次维持0.10～0.20秒，脉冲到期清零后冷却2～4秒。每个环境独立计时；出生/复位后先等待冷却，不在刚出生时立即推。
复用本机MJLab标准`apply_body_impulse`的时序、持续时长、复位和原生外力写入；本机接口只支持各轴同范围，薄适配层仅把采样Z轴上下界设为0。
保留标准原生外力可视化接口。不是给根部加支撑控制，也不改写速度制造假推力；没有LQR、额外竖直力、偏移施力点或扰动课程。
新日志隔离：`logs/rsl_rl/wangyu_moon_stand_push_v8/`，默认回放不回退旧v4模型。
TensorBoard新增`pelvis_push_force_N`与`pelvis_push_active_fraction`，记录实际原生外力大小和受扰动占比。
本次只完成代码与CPU事件/复位测试，没有运行v8训练、仿真或效率对照；随机扰动有望增强恢复能力，但不保证加快收敛。

下面保留历史版本与LQR诊断说明，当前主线不使用LQR。

历史平地LQR诊断入口：`stand-lqr` / `stand-lqr-flat` / `stand-v7` 指向独立任务 `Wangyu-Moon-Stand-LQR-Flat-v7`。
使用真正的MuJoCo无限plane（不是全零高度场），摩擦与增益准备平面一致；23轴、1/6G、默认8192环境。
从已有控制器的承重平衡姿态和配套PD目标复位，速度为0，不使用15mm离地间隙、随机位置/yaw或推力。
显示保留蒙皮和阴影。旧月面LQR v5改用 `stand-lqr-terrain` / `stand-v5`，其他v4入口不变。

```bash
./run.sh stand-lqr --agent zero                  # 先看纯LQR，RL残差为0，无需训练
./run.sh stand-lqr --agent zero --evaluate --num-envs 64 --seconds 20
./train.sh stand-lqr --agent.max-iterations 100  # 平地LQR+RL残差；独立v7日志
./run.sh stand-lqr                              # 有v7结果后，加载最新v7策略
```

v7日志：`logs/rsl_rl/wangyu_moon_stand_lqr_flat_v7/`。月面v5策略不能作为平地v7结果加载。
保留接触/偏离门控，不通过强行把权限置1来伪装控制成功；平衡点附近预计应有高接入权限，须实际运行验证。
新增平地入口目前只通过静态配置、模型编译和CPU数学/复位检查，未追加平地训练或仿真。

v7修正扰动时的参考切换：目标为 `ctrl0 - smoothstep(authority) * K * dx`，不再随权限退回源姿态零目标。只平滑权限映射，不对状态或整条命令加低通延迟；最终仍做物理限幅。旧平地v6保留为`stand-v6`，策略不能混用；已有LQR增益可复用，无需重新求解。该修正尚未实际做扰动仿真，不能宣称抖动已消除。

原无扰动对照：23轴月面站立 `Wangyu-Moon-Stand-Balance-v4`，8192环境，145维观测。
v4和下述v5均增加标准MJLab力矩项：`joint_torques_l2`，`-2e-6 * sum(tau_actual**2)`。
读取原生执行器实际输出，不是策略动作或请求力矩。当前23个执行器均为单位传动比关节执行器，故单位为Nm²。
例如23轴各输出10Nm时，该项惩罚率为-0.0046/s；TensorBoard同时记录奖励与`joint_torques_squared_Nm2`指标。
新增腰部Yaw（±40°）及左右踝Roll（±30°），头和手腕固定；源FBX初始姿态、蒙皮、质量、出生与地形设置沿用。
关节偏离奖励：`-1.0 * mean((q - q_initial)**2)`，23轴取均值，单位rad²，q_initial为源模型初始关节角（当前均为0）。
例如全部关节RMS偏离0.1rad时，额外惩罚率为-0.01/s；仍按控制步dt累计，允许恢复平衡所需的适度动作。
TensorBoard记录`joint_pose`奖励和`joint_pose_mean_squared_rad`指标。
日志隔离在`logs/rsl_rl/wangyu_moon_stand_balance_v4/`，20轴检查点不兼容，需新训。

```bash
./train.sh --agent.max-iterations 2000
./run.sh                         # 默认加载v4最新结果
./train.sh stand-v3              # 保留旧20轴实验
./run.sh stand-v3                # 保留旧20轴回放
```

可选实验：`Wangyu-Moon-Stand-LQR-v5`，8192环境、248维观测、23维RL残差，独立目录`logs/rsl_rl/wangyu_moon_stand_lqr_v5/`。v4仍作为对照。

```bash
./train.sh stand-lqr-terrain --agent.max-iterations 2000
./run.sh stand-lqr-terrain                       # 最新v5策略 + 该次训练的LQR快照
./run.sh stand-lqr-terrain --agent zero          # 仅LQR基础控制，RL残差为0
```

首次训练若缺少`outputs/lqr/stand_lqr_v1.npz`，入口先运行一次CPU准备：
原生`mj_inverse`搜索局部平衡点（检查全部6维浮动根残差、双脚承重及力矩/目标限幅），
原生`mjd_transitionFD`线性化，按两次物理步合成为60Hz控制模型，Riccati求解，并做5个各5秒的有界平地检查（含实际源姿态+15mm离地初始化，不能只验证参考点本身）。
这些都是**用户执行入口时的物理计算**；代码交付时没有运行准备、训练或仿真。
任一检查失败均停止，不保存无效增益、不静默回退。已有文件不会自动覆盖；需要重建时显式执行：

```bash
~/dev-envs/robotics-mjlab-py313/bin/python scripts/prepare_stand_lqr.py --force
```

控制为位置PD目标：`baseline + scale * clip(residual, -2, 2)`，最终按物理范围与±0.5rad交集限幅。
`scale = 0.05 + (1-authority)*0.20`：LQR完整接入时为0.05rad/unit，关闭时恢复v4的0.25rad/unit，防止无辅助的复杂地形只有±0.1rad纠正权限。
LQR为局部平地双脚支撑增益；不冒充各种地形的通用控制器。双脚承重不足、偏离参考姿态或坡度增大时降低权限：坡度≤3°为完整坡度权限，≥12°关闭基础LQR；无支撑或单脚支撑时关闭。关闭后仍有源初始姿态PD和RL残差。
保留源FBX姿态/原15mm安全出生间隙，**没有把全部随机地形出生状态宣称为静态平衡点**。接触建立阶段交给PD/残差，LQR按实际承重接入。
LQR不惩罚或追踪出生XY位置/yaw：线性模型剔除水平平移与yaw位置的规范自由度，保留高度、倾斜、关节偏差和速度；原1m安全活动范围终止保留。
原生MuJoCo Warp四元数差计算与`mj_differentiatePos`约定一致。训练循环只做批量坐标变换与矩阵乘法；actor/critic共享一次观测结果，动作处理复用相同状态的基础指令，无逐环境CPU动力学或在线Riccati求解。
RL均值输出层从0开始，初始std=.15；奖励保持v4，动作变化项改为最终合成目标差（按0.25rad归一化），力矩项自然覆盖LQR与RL共同产生的实际力矩。
新增指标：`lqr_authority`、`target_clipped_env_fraction`（基础或最终目标发生裁剪的环境比例）、`rl_residual_rms_rad`。

每个v5运行在`lqr/controller.npz`保存控制器快照，并将SHA256写入检查点。回放自动绑定该快照，拒绝用其他增益加载。
恢复v5训练时设置`WANGYU_LQR_FILE=/完整运行目录/lqr/controller.npz`，再使用MJLab原有resume/load参数；不把重新求得的增益混入旧策略。
导出的神经网络只包含RL残差，不包含外部LQR，部署时须一起携带。
目前只完成代码、模型静态编译和CPU数学检查；**增益尚未实际生成，稳定性、地形适应和学习提速均待用户运行验证**。

以下为历史版本说明，其中20轴/v3默认行为已由上述v4替代。`./run.sh --physics`静态/被动预览也恢复为23轴；旧20轴XML保留供历史任务加载。

# 望宇月面站立 · MJLab

## 当前默认：稳定控制 v3（代码已更新，未实跑）

```bash
./train.sh                    # v3，8192环境，从零训练；由用户执行
./run.sh                      # 只选v3最新非smoke运行的最大已保存编号
./run.sh --num_envs 8          # 仍为8个独立仿真世界；原生窗口逐个切换查看
./train.sh stand-terrain-v2    # 显式保留旧月面v2
./run.sh stand-terrain-v2
./run.sh stand-flat           # 旧平地v1
```

任务 `Wangyu-Moon-Stand-Balance-v3`，日志 `logs/rsl_rl/wangyu_moon_stand_balance_v3/`，观测136维。旧v2的97维网络不兼容，不自动迁移、不回退旧模型、不覆盖旧日志。新版本尚无模型时run明确报错。训练仍复用共享环境、8192×32、120Hz物理与60Hz控制，不自行启动训练或性能基准。

**重构依据。** 旧v2观测没有直接给策略总质心运动、角动量和脚载荷；仅把速度压到0不能区分静止站立与刚开始失稳，0.05m/s指数尺度也会在实际0.2m/s附近几乎饱和。旧缓存还将forward前奖励状态复用于forward后的观测。v3将reward与observation分为不同phase缓存，保持原生状态更新时间语义；不在每物理子步增加额外forward。

**原生信号。** 根部和躯干的framequat/frameangvel、骨盆原点framelinvel、双鞋底framepos/framelinvel/touch，以及Pelvis整个子树的subtreecom/subtreelinvel/subtreeangmom。优先使用MuJoCo/MuJoCo Warp已支持的传感器，不自行质量加权或重算动力学。骨盆速度来自骨盆原点site，和骨盆惯性COM速度、全身COM速度明确区分；H为全人体绕总质心的世界角动量，包含头盔与背包，不包含地形。

136维观测由关节位置/速度40、根部与躯干重力方向6、身体角速度6、骨盆和COM速度6、COM相对支撑点3、总角动量3、足相对COM位置6、足速度6、接触标志2、归一化法向载荷2、足底离地高度2、出生点偏移2、根部离地高度1、倾斜方向变化率4、预测支撑偏移2、上一动作20、局部地形高度25组成。使用当前状态，所有世界向量按重力水平朝向表达；倾斜变化率由方向向量与原生角速度的旋转关系得到，无差分历史尖峰。预测支撑偏移采用LIPM近似c_xy+sqrt(h/g)*v_xy，仅作提前感知的观测/诊断，不是MuJoCo原生的精确稳定裕量，不向物理施力。

**固定奖励权重（原始每秒量，RewardManager按控制dt积分）。**

| 分项 | 定义 | 权重 |
|---|---|---:|
| alive | 未触发失败，正常超时仍算存活 | +0.20 |
| pelvis_velocity | 骨盆原点三轴线速度平方和，(m/s)² | −4.0 |
| centroidal_angular_momentum | 原生全人体H三轴平方和，(kg·m²/s)² | −0.002 |
| gravity_tilt | 骨盆和躯干重力倾角平方的均值，rad² | −2.0 |
| action_change | 归一化位置动作差平方均值 | −0.002 |

未使用质心速度作为奖励，也没有恢复旧综合stability指数。示例：骨盆速率0.20m/s、H模长5kg·m²/s、骨盆/躯干倾角10°且其余项为零时，净奖励率约−0.071/s；若变为0.02m/s、H=1、倾角3°，则约+0.191/s。它只是权重预算例子，不是实际未训练分布；历史中没有H实测量，不能保证每次随机初始化或每轮奖励的正负。权重固定，没有按迭代增加alive或奖励偏置来伪造曲线。新TensorBoard指标 `Rollout/mean_reward_rate`、`Rollout/mean_step_reward`、`Rollout/positive_reward_fraction` 给出本轮真实符号变化，不额外打印Rollout行。

控制仍是20轴隐式位置PD，但v3动作缩放0.25rad、位置目标上限±0.5rad并与各关节物理限位取交集，避免v2仅±0.15rad纠偏范围过小。无额外根部支撑力，不修改质量、惯量或关节物理限位。actor/critic改为128×128并开启观测归一化，初始动作std0.25，gamma0.995以覆盖月面较慢的倾倒过程。

**有限地形与初始化保护。** 对照zbot_exoarm_moon的地形复位顺序：每个有效地形锚点周围XY±1.25m、yaw±π，旋转双脚后贴地、零初始速度；取全部50个鞋底采样点所需最高高度，加15mm间隙。先在启动时核对实际MuJoCo碰撞高度场的位置、尺寸、旋转、401×401数据与采样表一致，核对鞋底碰撞掩码。出生池及抖动后位置距地图边缘至少20m；每批reset检查全部环境的脚底采样点在图内且有限、最小间隙≥14.5mm、最大离地间隙≤20cm，不合格直接报错。该检查每个reset批次只回读一个布尔量，不逐世界循环；首次复位打印通过世界数、X/Y范围和脚底间隙。所有8192个独立世界使用同一完整高度场，不是只有某些世界有地面。偏离出生点1m、接近地图边缘、躯干倾角45°、骨盆倾角30°或低于地面高度下限会终止复位。

**显示。** v3显式开启阴影，4096阴影图、局部shadowclip、斜向投影光与弱补光；主光使用原生TRACK模式随人物平移、保持世界方向，避免随机到远处后脱离局部阴影范围。保留近裁剪与骨盆跟随配置。只影响回放，不改训练物理。没有启动窗口核验，因此不能声称脚底阴影效果已经实测。

`tests/test_balance_math.py`提供不构造环境的CPU数学回归，覆盖观测维度/离地有限值、原生骨盆速度与H量纲、世界平移不变性、倾斜变化率符号和示例奖励预算。本次只做语法编译、shell检查及本机API/时序源码核查，没有运行这些测试、仿真、训练或8192环境基准。**新增自检不等于8192世界已实测通过；新配方也不等于已学会稳定。**


## 历史：月面随机站立 v2

```bash
./train.sh                  # Site20站立，默认8192环境；由用户自行运行
./run.sh                    # 最新非smoke运行中编号最大的已保存模型，带蒙皮
./run.sh --num-envs 4        # 同一选模规则
./run.sh --evaluate --num-envs 16 --seconds 20
./train.sh stand-flat       # 显式保留旧平地v1
./run.sh stand-flat
./run.sh --physics          # 资产预览，不加载策略
```

新任务 `Wangyu-Moon-Stand-Terrain-v2`，日志 `logs/rsl_rl/wangyu_moon_stand_terrain_v2/`。`stand`、`stand-terrain`为新默认任务别名。默认从零训练；97维观测与旧平地72维模型不兼容，不自动回退或迁移旧结果。run打印任务、日志根目录、最新运行目录、检查点绝对路径及文件名、环境数、设备、重力和地形信息；新任务没有已保存模型时明确报错。

使用现有Site20/Leibnitz beta plateau真实DEM的中央2km×2km区域，5m原始分辨率、401×401原生MuJoCo高度场，保留米制尺度与起伏。仅去除中心海拔，数据约776KB；不依赖运行时Isaac/USD或原工程绝对路径。脚以下的三角形插值与MuJoCo高度场的SW→NE对角线一致，不用双线性插值代替碰撞表面。该DEM不包含5m以下的真实碎石细节，没有把合成噪声冒称实测月面。

每回合各环境独立随机选择出生XY，格心周围±1m抖动，yaw均匀±π；出生池覆盖不同朝向的真实坡面和起伏地带，邻近坡度上限15°，避开边缘及明显不适合站立的陡坡。陡坡仍存在于碰撞地形中，不是把地形压平，也不宣称任意悬崖可站立。无课程。初始关节姿态沿用3D模型，按yaw旋转双鞋底各25个采样点，取所需的最大骨盆高度并加1cm间隙；不只读取骨盆正下方的高度。刚性平鞋不能在所有坡面同时贴合，后续接触与姿态由物理解算及策略调整。

站立目标是每次出生的根部位置，不再固定世界原点；XY误差在出生yaw坐标系表达，竖直仍按世界重力，随机朝向不会被误判倾斜。跌倒高度使用骨盆相对当前脚下地面的高度，不用绝对世界Z。观测保留72维本体状态，增加身体水平朝向下5×5局部高度采样，共97维；高度相对当前位置地面，避免绝对海拔干扰。

沿用20轴、固定头/手腕、精确1/6G、PD与站稳/动作变化奖励，只允许鞋底碰地，双脚互碰排除。地形版本单独预留128接触、512约束/世界，不沿用平地8接触假设。原生物理图、张量图、批量回合日志、无蒙皮训练/有蒙皮回放继续保留；高度数据、出生池、鞋底采样与扫描网格只在初始化上传GPU，复位不逐环境循环或回传CPU。

本次仅进行了资产数据转换、代码静态审查、Python语法编译与shell语法检查；未启动仿真、训练、评估或8192环境性能基准。因此地形高度查询与实际接触的运行时一致性、显存/接触容量、训练吞吐和站立成功率仍需用户运行后确认。旧平地的20秒成功数据不代表新地形结果。


复用 `~/dev-envs/robotics-mjlab-py313`（MuJoCo/Warp 3.11），没有升级共享环境。

```bash
./train.sh --jump --env.scene.num-envs 4 --agent.max-iterations 2
./run.sh --help
```

`./run.sh --show-model` 打开望宇完整外观的 MuJoCo 静态预览，可用鼠标旋转、缩放。预览资产位于 `assets/wangyu_visual/`，从已有 FBX 烘焙生成，尚未与训练代理骨架绑定。
旧跳跃任务需显式使用 `./run.sh --jump --agent zero --num-envs 1 --viewer native`（显示旧训练代理模型）。

`src/moon_task.py`：独立任务 Wangyu-Moon-Jump-v0。
`assets/wangyu_proxy.xml`：20主动轴、浮动根、固定头/手腕，月球重力1.625m/s²。尺寸、质量、限位当前为工程代理值，尚未按望宇FBX完整标定，尚未接入望宇蒙皮或外骨骼。
奖励是初始跳跃探索奖励，不是原Apollo模仿配方；短训只验证链路，不代表学会跳跃。仅脚碰地；后续脱困必须单独扩展碰撞。

原项目代码/训练/资产未覆盖。当前不需要迁移大型贴图、历史检查点或Isaac环境。

## 望宇原生物理资产（20轴）

`assets/wangyu_physics/wangyu.xml` 是纯 MuJoCo 原生几何的机器人资产，无 mesh 或贴图依赖。`scene.xml` 为带平地的独立预览场景。

- 13个body节点，头部固定在躯干上，手与前臂合并；20个驱动轴、6个浮动根自由度。
- 腰2、髋6、膝2、踝2、肩6、肘2；无腰部旋转、踝侧倾、颈部或手腕自由度。
- 月球重力1.625 m/s²；仅双鞋底碰地，无自碰撞，无外骨骼。
- 全部关节原点及肢段方向沿用望宇FBX的初始姿态，保留自然外展的双臂、前屈的肘部与原始腿部姿态；仅将骨盆XY居中。躯干、背包及半径等仍为工程近似。
- 126.8 kg及分段质量沿用阿波罗工程估计，不是望宇实测；背包暂取25 kg并计入躯干，惯量由新原生几何重新计算。
- 多轴关节采用MuJoCo顺序铰链，不能直接套用阿波罗USD旋转向量动作。

```bash
./run.sh --physics                  # 同上；空格启停被动仿真，R复位，P切换蒙皮/原生几何
./run.sh --physics --simulate       # 直接开始无控制器的被动仿真
./run.sh --physics --agent zero --num-envs 1 --viewer native
./train.sh --physics --env.scene.num-envs 4 --agent.max-iterations 2
```

新任务 `Wangyu-Moon-Jump-Physics-v1` 使用独立日志 `wangyu_moon_jump_physics_v1`，保留原任务和旧检查点。奖励仍为原起步探索配方，没有迁移阿波罗模仿训练。

重建：使用共享环境Python运行 `scripts/build_wangyu_physics.py`；所需骨架测量及阿波罗参考快照已保存在资产目录。`scripts/extract_wangyu_physics.py` 可在Blender内从源FBX重新提取测量。
验证：`MUJOCO_GL=egl <共享环境Python> scripts/validate_wangyu_physics.py`，报告及预览保存至 `outputs/wangyu_physics/`。验证覆盖惯量、关节限位、驱动方向、静态支撑力矩及2秒被动落地；不代表已学会站立或跳跃。

## 随动物理蒙皮

`./run.sh --physics` 显示20轴物理骨架驱动的望宇蒙皮；`./run.sh --physics --proxy` 显示原生几何。窗口内按P切换，空格启停无控制器的被动仿真，R复位。

`assets/wangyu_skin/skin.xml` 是MuJoCo原生线性混合蒙皮，保留FBX顶点权重并汇总到13个物理body；手指、手腕合并到前臂，头部固定。物理零位与FBX初始位姿一致，蒙皮使用同一套源姿态绑定；初始顶点仅随骨盆XY居中整体平移。运行时直接使用物理body位姿，不新增质量、惯量、碰撞或自由度。当前是降阶骨架的显示近似，极端姿态仍可能出现局部穿插或布料变形；不模拟航天服布料动力学。

独立场景为 `assets/wangyu_skinned.xml`；无地面的MJLab实体入口为 `assets/wangyu_skinned_robot.xml`。原生物理任务的play配置使用蒙皮，train配置继续只加载原生几何。MuJoCo 3.11的attach未自动前缀化skin.material引用，因此任务spec函数按固定实体名robot/适配；未修改共享依赖。

重建蒙皮：`blender -b --python scripts/export_wangyu_skin.py -- <源FBX路径>`。核验：`MUJOCO_GL=egl <共享环境Python> scripts/validate_wangyu_skin.py`。`outputs/wangyu_physics/skin_validation.json`记录参数一致性、240步物理对照和20轴逐轴变形检查；`skinned_preview.png`、`skinned_bend.png`为源初始位姿及屈膝视觉检查图。

初始位姿更新：20个关节在数值0时表示源FBX姿态，不再表示手臂垂直贴身。现有关节限位按这个零位表达，属于工程范围，尚未按人体绝对解剖角度重新标定。`skin_validation.json`同时核对13个body的源关节原点与43,786个蒙皮顶点的初始位姿一致性（允许的最大误差1微米）。

## 最小原地站立实验

任务 `Wangyu-Moon-Stand-v1`，独立日志 `logs/rsl_rl/wangyu_moon_stand_v1/`，不改跳跃实验。默认8192环境、200轮PPO、每轮32控制步；120Hz物理、60Hz控制，每轮262144个环境控制步、64次物理子步调用。使用共享环境与1个CPU Torch线程。

```bash
./train.sh stand-flat                                      # 默认8192环境，从零训练
./train.sh stand-flat --agent.max-iterations 50             # 有界短训
./run.sh stand-flat                                        # 最新非smoke运行的最大已保存编号，带蒙皮
./run.sh stand-flat --evaluate --num-envs 16 --seconds 20    # 独立确定性评估，结束后关闭环境
./run.sh stand-flat --agent zero                           # 无学习的PD基线，不伪装训练结果
```

物理：重力严格为9.81/6=1.635m/s²；沿用望宇FBX初始姿态、20轴、浮动根、固定头/手腕、原生质量惯量；仅鞋底与平地碰撞，额外排除双脚互碰，无根部外力、固定根或支架。两块盒形鞋底最多8个接触，分配16接触/世界、96约束/世界并保留容量余量，NXN广相。训练不加载蒙皮，回放加载原生MuJoCo skin。

控制：20维初始位姿附近的关节位置目标，动作乘0.10rad后按关节范围与±0.15rad交集裁剪；隐式PD，下肢Kp300/Kd25、腰臂Kp100/Kd10，下肢力矩限制180Nm、腰臂60Nm。不是纯力矩策略。显式reset_scene_to_default按每个环境原点复位，恢复源初始姿态，不使用随机化、推力、地形课程。

奖励只有稳定度与轻微动作变化惩罚。稳定度为初始位置/高度、直立程度、身体线角速度、关节运动和姿态偏差的联合指数分数；动作变化权重−0.002。倾斜超过30°或骨盆低于初始高度70%时终止，正常回合10秒超时。

效率：复用R3的FixedTensorGraph方法，将72维观测和稳定奖励合并为固定形状GPU张量计算；每控制步共享结果，完整/局部reset均失效缓存，图输出使用独立存储。GPU回合日志每轮批量读回最近100个已完成回合，保留时间顺序与滑动窗口；物理CUDA Graph由MJLab原生启用，保留上游NaN检查，不修改共享库。回退对照：`WANGYU_TENSOR_GRAPH=0`、`WANGYU_FAST_LOG=0`；不改变物理/奖励/采样量。

已验证：`scripts/check_stand.py`检查图与eager一致、复位、日志顺序、动作裁剪、碰撞及重力；`scripts/benchmark_stand.py`在同进程同配置交替测量8192环境的eager/张量图采样。报告位于 `outputs/stand/`。其中pure采样不含策略、PPO与日志，不可与完整训练FPS混为一谈；Torch显存字段也不含Warp分配。

首次50轮/13107200样本短训已生成model_49。固定初态确定性评估16环境×20秒均无跌倒，倾角RMS约1.01°，根部速度RMS约0.0064m/s，最大XY漂移约2.3cm。评估剔除最初1秒计算RMS，但全程计入跌倒与最大漂移；门槛为无跌倒、倾角RMS<3°、根速度RMS<0.03m/s、关节速度RMS<0.1rad/s、最大漂移<5cm。仅是固定初态有限时长仿真证据，未验证扰动或实机。

开发参考：本地zbot_rl_mjlab_R3的cuda_graph.py、benchmark_collection.py与项目经验；[mjlab-skillkit](https://github.com/cmjang/mjlab-skillkit)用于原生任务结构、显式超时/复位、训练/回放分离及API核对。后者为开发参考，未安装为全局技能，其API示例须以本机MJLab1.6.0/MuJoCo-Warp3.11.0真实签名为准。

月面回放黑屏修正：play配置将scene.extent固定为max(4m,1.5×相机距离)，当前5.25m，近裁剪面约1.05cm；显式跟随robot/Pelvis并启用相机补光，避免2km地形自动extent把近处人物裁掉。远裁剪约5.25km。仅修改显示配置，不改变训练物理；已做静态语法检查，未启动回放验证。

回放日志：默认将MJLab/Warp导入、环境构造和网络加载等初始化输出保存到 `/tmp/wangyu-run-init-*.log`，终端保留选模路径、运行参数和日志位置；进入viewer循环后恢复正常输出，异常traceback仍显示。`./run.sh --verbose-init` 或 `WANGYU_VERBOSE_INIT=1 ./run.sh` 可显示完整初始化日志。训练日志不受影响。

重力倾角诊断（平地v1与月面v2共用）：新增 `Episode_Metrics/gravity_tilt_mean_deg` 和 `Episode_Metrics/gravity_tilt_peak_deg`，分别记录每回合的平均倾角与峰值倾角，单位度；MJLab日志对当批完成回合取均值，因此peak字段是完成回合峰值的均值，不是全训练绝对最大值。倾角使用已有身体坐标系重力方向，以atan2计算，对随机yaw不敏感；GPU上计算和汇总，不增加网络观测维度、不改变奖励或终止条件。下次启动训练加载，历史日志与运行中进程不修改。本次仅做代码及静态语法检查。

性能只读检查：现有同为8192×32、dt1/120、decimation2的日志，平地最近20轮采样中位0.241s/PPO0.057s，月面采样0.316s/PPO0.056s；月面运行配置metrics为空，故该次降速早于倾角日志。任务、接触容量、地形采样与复位工作量不同，不作逐核归因。已将倾角并入原张量图，mean/peak共享一次计算；跌倒检查的固定批量地形高度查询也使用固定形状图，动态reset查询不改。未启动训练/仿真/基准，优化收益未实测，报告见outputs/stand_terrain/performance_review.json。

月面v2奖励更新：保留综合stability，同时新增独立 `gravity_tilt` 惩罚，原始值为相对世界重力竖直方向的倾角平方θ²（rad²），权重−1。该角度在既有张量图内计算，与倾角日志共享，不增加97维网络观测。日志 `Episode_Reward/gravity_tilt` 为负贡献；权重是未训练标定的初始值，后续依据实际训练日志调整。只作用于月面任务，旧平地奖励不变。旧月面检查点仍可回放，但不因此视为已接受新惩罚训练；新配置在下次训练启动时生效。未启动训练或仿真，仅做静态语法检查。

月面stability定义更新：使用原生MuJoCo `subtreelinvel` 传感器，挂到浮动根Pelvis，直接读取整个13刚体人体子树（含头盔、背包）的总质心世界线速度；静态地形不属于该子树。传感器启用MuJoCo Warp的原生subtree velocity计算，不使用骨盆速度代替，也不自行逐刚体加权或CPU差分。`stability=exp(-||v_COM||²/0.05²)`，权重1，包含XYZ三轴。新增 `Episode_Metrics/com_speed_m_s`（回合平均总质心速率）。独立gravity_tilt=−θ²、动作变化惩罚、跌倒判据和97维观测保留；原综合位置/高度/关节/角速度项不再进入月面stability，旧平地定义不改。仅代码及语法核查，未运行仿真或训练。

望宇贴图与蒙皮静态检查：本次修正了源FBX/Blender UV与MuJoCo图片原点的V方向差异，三个PNG纹理统一设置`vflip=true`；UV和蒙皮权重本身不翻转。肩部红条、膝部护片及背包标识已与FBX参考图核对。

```bash
MUJOCO_GL=egl ~/dev-envs/robotics-mjlab-py313/bin/python scripts/audit_wangyu_skin.py
```

该工具检查实际`robot/`命名空间下的23轴、9块蒙皮、权重和材质，并静态渲染正面/背面/弯曲姿态，不推进物理、不启动策略。结果在`outputs/skin_audit/`。物理参数不变，已有策略可继续使用；关闭并重启回放以重新加载纹理。


### 训练采样计算优化（2026-11-15）

参考本机 MuJoCo `doc/skills/accelerated/SKILL.md` 与 MJWarp 性能文档。
可选 `WANGYU_REWARD_FAST=1`（默认0，用户报告退化后已回退）：balance_mdp 的 signal 奖励/统计只计算原有13列结果，不再构造145维观测或查询地形；独立 reward CUDA Graph 保留固定输入、输出拷贝和原 pre/post-forward/reset 代次缓存。完整观测仍查询地形，扫描中心复用同阶段根部高度。无共享依赖修改。

默认 v15 每次奖励包少做25点扫描、扫描中心1点、根部与双脚3点高度查询；观测再省重复中心1点。8192环境下一次奖励包少查询237568个点，这是代码工作量减少，不是实测吞吐增幅。146维观测、奖励权重、8192环境、32步rollout、物理步长/decimation、PPO、扰动及异常保护不变。CPU float32/float64逐值等价、部分复位/阶段刷新和扫描中心复用测试通过；未启动GPU训练/基准，CUDA Graph运行尚待验证。

奖励完整路径对照入口（由用户执行）：

```bash
WANGYU_REWARD_FAST=0 ./train.sh stand
```

该开关仅回退奖励包优化；`WANGYU_TENSOR_GRAPH=0` 可另行关闭项目张量图，不能关闭MJLab原生物理图。接触容量和求解器不因未测的性能假设下调。

2026-11-15性能退化反馈：11:52运行前6轮采样约2.16–2.38s、学习约.14–.15s；同配置11:44运行同期约.68–.70s/.057–.067s（不含首轮）。尚未确认因果。奖励分离和扫描中心复用已默认关闭，显式WANGYU_REWARD_FAST=1才启用；后续运行启动打印并保存compute_options.json。未重新训练验证恢复。

### 月面站立先验 + 质心动量与跳高两项指数奖励（hop-moon v2）

用户最新取消起跳门控。动量项upward_momentum：每控制步读取原生Pelvis SUBTREELINVEL整个人体COM世界速度，沿实际重力反向投影，乘原生Pelvis body_subtreemass得到p_up。目标p_target=M sqrt(2|g|h)，x=clip(p_up/p_target,0,1)，r=expm1(5x)/expm1(5)。不检查脚接触/是否离地/出生历史，不使用状态机、峰值或动量积分；接地向上运动也能得分，静止/水平/下落为0。RL回报按原RewardManager逐步乘dt再累计，这已取消原来的“每次离地仅结算一次”。物理动量仍是瞬时量，不跨帧或跨跳跃累计。

默认h=.25m；原生质量126.8kg、|g|=1.635m/s²，对应目标速度.904157m/s、目标动量114.647116kg·m/s。h是理想自由飞行COM沿重力反向增高的标定值，尚有接触时达到该动量只说明向上运动，不能保证离地或实际达到该高度。固定质量下此归一化比例等价于v_up/sqrt(2gh)。本机MJLab gravity_vec_w写死-Z，这里使用native model.opt.gravity方向；无直接子树线动量传感器，只将原生总子树质量乘原生子树速度，不自写逐体加权。

权重以src/jump_task.py为准，用户当前设置50.0，本次不改变。146维观测、原全部奖励与权重、23轴原生物理/PD、8192环境/10秒回合、出生/异常保护不变。动量项仍无起跳门控；新增高度项见下文，仅维护本回合首次触地COM基准。评估报告原生COM升高，不把它当作鞋底间隙或落地成功。

```bash
./train.sh hop-moon --jump-height 0.25 --agent.max-iterations 100
./run.sh hop-moon --jump-height 0.25
./run.sh hop-moon --jump-height 0.25 --evaluate --num-envs 16 --seconds 30
```

新日志upward_momentum_exp1_k5_h{h}加_no_push或_forceN_torqueM，与旧takeoff_momentum_height1和upward_velocity_only1隔离。站立actor及normalizer严格加载146维，新任务critic/optimizer/iteration全新；旧148维不可裁剪加载。旧146维v2可显式resume，奖励已变，新日志隔离。默认无随机扰动，只有--pushes或显式--push-force/--push-torque开启，--no-pushes优先，红色箭头仍关闭；默认stand任务不变。仅CPU静态/公式/无门控回归/help检查，无新训练/仿真。

当前指数曲线的k由src/jump_settings.py MOMENTUM_EXPONENT设置（默认5），奖励权重仍由src/jump_task.py设置（用户当前50）。静止严格0、目标及以上1，不引入Gaussian的静止正偏置。10%目标动量得.00440、50%得.07586、80%得.36359，显著压低反复低动量的收益。形状变化不修改其他奖励/观测/物理/默认无扰动，RL回报仍逐步累计，不能数学上保证永不通过重复小跳获益。指数参数限定(0,50]以保证float32数值安全。

用户反馈最新模型已跳但高度不够。读取已有15:26-13的300代日志：末10代Episode_Reward/upward_momentum=7.33467（权重50）、COM平均总速.30333m/s；日志没有实际COM跳高/离地峰值，不能从平均有符号竖速推高度。已有跳跃模型可用--resume-from指向原upward_momentum_dense1_h0.25_no_push/2026-11-15_15-26-13/model_299.pt继续学习指数奖励，新日志隔离。无--resume-from时默认仍从站立先验初始化。只做已有日志读取与CPU静态曲线/模型/配置测试，未训练仿真，不能声称跳高已改善。


用户新增跳高指数奖励：jump_height权重10（src/jump_task.py），高度指数HEIGHT_EXPONENT=5（src/jump_settings.py），目标共用--jump-height默认.25m。r_h=airborne*expm1(5*clip(h_COM/h_target,0,1))/expm1(5)。h_COM来自原生Pelvis SUBTREECOM世界位置，相对本回合首次原生鞋底接触快照沿实际重力反向的升高；基准整回合固定，不随蹲起移动。只在当前双鞋均无原生接触时给高度分（found/dist<=2mm/finite pos），不恢复动量门控。初始空隙没有基准所以0，接地伸直或单脚接触0，COM到半目标高度得.07586、达到目标得1；下降过程中仍有相应高度分。高度是COM相对首次触地位置的位移，不是鞋底离地间隙、也不是每次离地瞬间重新标定的飞行高度；地形移动可能影响基准解释。原生接触/COM读同一pre-reset阶段，不额外forward，不作新动力学或逐体质心计算。部分reset只清对应基准；actor/critic观察不更新它，146维/物理/其他原奖励不变。

动量指数项保持当前权重50、指数5和无门控。新日志momentum_height_exp1_km5_kh5_h{h}加扰动后缀隔离，默认无扰动/无红箭头。新增jump_com_rise_m、jump_com_rise_peak_m、jump_airborne_com_rise_peak_m、jump_height_airborne_fraction；峰值为日志的每回合峰值再聚合，不当成所有轨迹的全局最大值。独立评估输出max_com_rise_m/max_airborne_com_rise_m。两个指数项RL回报均按dt逐步累加，不能宣称绝对杜绝重复获益。默认仍从站立先验初始化，继续已有跳跃需--resume-from。已有25项CPU静态/方向/原生COM平移/出生空隙/接地伸展/部分复位回归通过，未运行训练仿真或宣称跳高改善。
