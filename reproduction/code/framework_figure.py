"""Original editable vector schematics; no learned or measured quantities invented."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Polygon,Circle
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'KBS_submission/figures'
OUT.mkdir(exist_ok=True)
NAVY='#233C59'; TEAL='#278B87'; VIOLET='#78648D'; GOLD='#B68042'
INK='#203247'; MUTED='#617181'; BORDER='#CFD9E1'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.2,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,
 'savefig.facecolor':'white'})

class Scene:
 def __init__(self,name,h):
  self.name=name;self.h=h;self.items=[]
  self.fig,self.ax=plt.subplots(figsize=(6.75,h/160),layout=None)
  self.fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
  self.ax.set(xlim=(0,1080),ylim=(h,0));self.ax.axis('off')
 def box(self,x,y,w,h,fill='white',stroke=BORDER,r=8,lw=.7):
  self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
      facecolor=fill,edgecolor=stroke,linewidth=lw))
  self.items.append(('box',x,y,w,h,'',fill,stroke))
 def text(self,x,y,s,size=8.2,color=INK,weight='normal',ha='left'):
  self.ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',linespacing=1.45)
  self.items.append(('text',x,y,0,0,s,color,weight,size,ha))
 def line(self,x1,y1,x2,y2,color=MUTED,lw=1.0,arrow=False,dashed=False):
  if arrow:
   self.ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,
       linewidth=lw,color=color,linestyle='--' if dashed else '-'))
  else:self.ax.plot([x1,x2],[y1,y2],color=color,lw=lw,ls='--' if dashed else '-',solid_capstyle='round')
  self.items.append(('edge',x1,y1,x2,y2,'',color,lw,arrow,dashed))
 def dot(self,x,y,color,r=4):
  self.ax.add_patch(Circle((x,y),r,facecolor=color,edgecolor='white',lw=.6))
  self.items.append(('dot',x,y,r,r,'',color,'white'))
 def save(self):
  import sys
  sys.path.insert(0,str(ROOT/'reproduction/code/figure_checks'))
  from audit_panel_alignment import require_matplotlib_panel_alignment
  self.fig.canvas.draw()
  require_matplotlib_panel_alignment(self.fig,json_out=OUT/f'{self.name}.alignment.json',strict=True)
  self.fig.savefig(OUT/f'{self.name}.pdf')
  plt.close(self.fig)

def mechanism():
 s=Scene('principle',440)
 xs=[12,284,556,828];cols=[NAVY,TEAL,VIOLET,GOLD]
 titles=['Predictive interface','Minimum repair set','Decision sets','Guarantee endpoints']
 for k,(x,c,t) in enumerate(zip(xs,cols,titles)):
  s.box(x,14,240,380,fill='#FAFCFD')
  s.box(x,14,240,52,fill=c,stroke=c,r=8)
  s.text(x+14,39,chr(97+k),9.5,'white','bold')
  s.text(x+40,39,t,7.7,'white','bold')
  if k<3:s.line(x+243,208,x+266,208,color=MUTED,arrow=True)
 # Interface: current prediction and two separately queried contexts.
 x=xs[0]
 s.box(x+21,91,198,52,fill='#EAF0F6',stroke='#BDCBDB')
 s.text(x+120,117,'Predictor in context D',7.6,NAVY,'bold','center')
 for y,label in [(179,'Current:  a, b'),(224,r'Append A = 0:  $q_0$'),(269,r'Append A = 1:  $q_1$')]:
  s.box(x+32,y,176,33,fill='white',stroke='#CAD6E1')
  s.text(x+120,y+16.5,label,8.1,INK,ha='center')
 s.line(x+120,145,x+120,174,NAVY,arrow=True)
 s.text(x+120,332,r'$m=(1-a)q_0+aq_1$',8.5,NAVY,ha='center')
 s.text(x+120,357,r'$m\ne b$: inconsistent updates',7.4,MUTED,ha='center')
 # Entire minimum face, rather than selecting one point.
 x=xs[1]
 s.text(x+120,104,'Preserve a and b',8.4,TEAL,'bold','center')
 s.text(x+120,138,r'$\min C(t)=|b-m|$',10,TEAL,ha='center')
 s.line(x+32,221,x+208,221,color='#B7C6CF',lw=1)
 s.line(x+58,221,x+172,221,color=TEAL,lw=7)
 s.dot(x+58,221,TEAL,5);s.dot(x+172,221,TEAL,5)
 s.text(x+58,250,'ℓ',9.2,TEAL,ha='center');s.text(x+172,250,'u',9.2,TEAL,ha='center')
 s.text(x+115,187,'Every minimizing joint',8.1,TEAL,'bold','center')
 s.text(x+120,289,r'$\mathcal{T}=[\ell,u]$',10,TEAL,ha='center')
 s.text(x+120,332,'Equal minimum cost\nDifferent dependence',8.1,INK,ha='center')
 s.text(x+120,367,'Binary interval / finite-label face',6.9,MUTED,ha='center')
 # Quantitative bounds from the exact two-candidate construction.
 x=xs[2];origin=x+59;scale=285
 s.text(x+120,104,'Sharp information ranges',8.2,VIOLET,'bold','center')
 for y,label,l,u,c in [(158,r'$A_1$',.029049405545331,.531004406410719,VIOLET),(216,r'$A_2$',0,.278071905112638,TEAL)]:
  s.text(x+21,y,label,8.2,INK)
  s.line(origin+scale*l,y,origin+scale*u,y,color=c,lw=6)
  s.dot(origin+scale*l,y,c,4.5);s.dot(origin+scale*u,y,c,4.5)
  s.text(origin+scale*(l+u)/2,y+26,f'[{l:.3f}, {u:.3f}] bits',7.5,c,ha='center')
 s.text(x+120,280,r'$W_1$: $A_2$ wins  /  $W_2$: $A_1$ wins',7.3,VIOLET,'bold','center')
 s.text(x+120,324,'Same interface\nTwo coherent worlds',8.0,INK,ha='center')
 s.text(x+120,367,'Strict / tolerant / possible',7.0,MUTED,ha='center')
 # Distinct assumptions rather than a causal ladder.
 x=xs[3]
 for y,title,subtitle,c in [(104,'LOCAL','Sharp information bounds',TEAL),(183,'GLOBAL','Compatible joint + regret',VIOLET),(262,'LOSS','Historical labels + fixed law',GOLD)]:
  s.box(x+20,y,200,65,fill='white',stroke='#CCD7DF')
  s.text(x+33,y+19,title,7.5,c,'bold')
  s.text(x+33,y+44,subtitle,7.0,INK)
 s.text(x+120,367,'Oracle MI: separate endpoint',6.8,MUTED,ha='center')
 s.text(18,422,'Blue: observed outputs   ·   Teal: repair geometry   ·   Purple: decisions   ·   Ochre: calibration assumptions',7.1,MUTED)
 s.save()


if __name__=='__main__':
 mechanism()
