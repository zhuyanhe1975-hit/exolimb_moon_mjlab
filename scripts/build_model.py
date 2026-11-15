from pathlib import Path
import xml.etree.ElementTree as E
r=E.Element('mujoco',model='wangyu_moon_proxy');E.SubElement(r,'compiler',angle='radian');E.SubElement(r,'option',gravity='0 0 -1.625',timestep='0.008333333333')
d=E.SubElement(r,'default');E.SubElement(d,'joint',damping='3',armature='.05',limited='true');E.SubElement(d,'geom',type='capsule',rgba='.85 .87 .9 1',friction='1 .01 .001',contype='0',conaffinity='0');w=E.SubElement(r,'worldbody');p=E.SubElement(w,'body',name='Pelvis',pos='0 0 .88');E.SubElement(p,'freejoint');E.SubElement(p,'geom',type='box',size='.12 .16 .10',mass='18');act=E.SubElement(r,'actuator')
def joint(b,name,axis,ran):
 E.SubElement(b,'joint',name=name,axis=axis,range=ran);E.SubElement(act,'motor',name=name,joint=name,ctrllimited='true',ctrlrange='-150 150')
def seg(parent,name,pos,end,radius,mass):
 b=E.SubElement(parent,'body',name=name,pos=pos);E.SubElement(b,'geom',fromto='0 0 0 '+end,size=str(radius),mass=str(mass));return b
body=seg(p,'Torso','0 0 .1','0 0 .4',.16,35);joint(body,'WaistPitch','0 1 0','-.5 .5');joint(body,'WaistRoll','1 0 0','-.3 .3');E.SubElement(body,'geom',type='box',pos='-.20 0 .20',size='.12 .20 .32',mass='25',rgba='.9 .9 .85 1');E.SubElement(body,'geom',type='sphere',pos='0 0 .53',size='.14',mass='6')
for side,y in [('Left',.13),('Right',-.13)]:
 thigh=seg(p,side+'Thigh',f'0 {y} 0','0 0 -.36',.085,11)
 for suffix,axis,ran in [('Pitch','0 1 0','-1.8 .4'),('Roll','1 0 0','-.5 .5'),('Yaw','0 0 1','-.5 .5')]:joint(thigh,side+'Hip'+suffix,axis,ran)
 shin=seg(thigh,side+'Shin','0 0 -.36','0 0 -.35',.07,7);joint(shin,side+'Knee','0 1 0','0 1.9')
 foot=E.SubElement(shin,'body',name=side+'Foot',pos='0 0 -.35');joint(foot,side+'Ankle','0 1 0','-.6 .6');E.SubElement(foot,'geom',type='box',pos='.07 0 -.08',size='.15 .075 .08',mass='3',contype='1',conaffinity='1')
 arm=seg(body,side+'UpperArm',f'0 {y*1.6} .35','0 0 -.28',.055,4)
 for suffix,axis in [('Pitch','0 1 0'),('Roll','1 0 0'),('Yaw','0 0 1')]:joint(arm,side+'Shoulder'+suffix,axis,'-1.5 1.5')
 fore=seg(arm,side+'ForeArm','0 0 -.28','0 0 -.30',.05,3);joint(fore,side+'Elbow','0 1 0','-1.8 .1')
p=Path(__file__).resolve().parents[1]/'assets/wangyu_proxy.xml';E.indent(r);E.ElementTree(r).write(p,encoding='unicode')
