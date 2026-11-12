# -*- coding: utf-8 -*-
"""混合型中心裂纹的上部子域源点分布示意图，仅依赖 numpy、matplotlib。
运行 python mixed_mode_crack_source_schematic.py，无需训练。
默认几何依据参考图片重绘：x∈[-2,2], y∈[-4,4], a=1, θ=30°。
源点使用上部子域多边形的等距外偏移边界；实际求解规则未提供，
本脚本的点坐标仅用于示意，不声称与求解器逐点相同。
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Arc

# ---------------- 可调整参数 ----------------
HALF_WIDTH = 2.5
HALF_HEIGHT = 4.0
CRACK_HALF_LENGTH = 1.0
CRACK_ANGLE_DEG = 30.0             # 相对水平轴的倾角，允许正负倾角
SOURCE_OFFSET = 0.28              # 源点边界到子域边界的垂直距离
N_SOURCES_PER_EDGE = 80           # 四条边总计320，排除角点
N_BOUNDARY_PER_EDGE = 200         # 仅为绘图配点参数，请按求解器设置修改
SHOW_SOURCES_PER_EDGE = 9
SHOW_BOUNDARY_PER_EDGE = 15
SHOW_ALL_POINTS = False
SHOW_BOUNDARY_POINTS = True
SHOW_TITLE = True
SHOW_ANGLE = True
SHOW_FIGURE = True
SAVE_FIGURES = True               # False时只显示、不保存
OUTPUT_DIR = None                 # 默认脚本目录/mixed_mode_source_figures
# --------------------------------------------
BLACK = '#252525'
GRAY = '#8C8C8C'
RED = '#BE4540'
BLUE = '#326A93'
STYLE = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
         'mathtext.fontset': 'stix', 'font.size': 11,
         'svg.fonttype': 'none', 'pdf.fonttype': 42}


def edge_points(vertices, count):
    if int(count) != count or count < 1:
        raise ValueError('每边点数必须为正整数。')
    t = np.linspace(0., 1., int(count) + 2)[1:-1, None]
    return [(1-t)*p + t*q for p, q in zip(vertices, np.roll(vertices, -1, axis=0))]


def display_subset(points, count):
    if int(count) != count or count < 1:
        raise ValueError('展示点数必须为正整数。')
    if count >= len(points):
        return points
    ids = np.rint(np.linspace(0, len(points)-1, int(count)+2)[1:-1]).astype(int)
    return points[ids]


def geometry():
    b, h, a, d = HALF_WIDTH, HALF_HEIGHT, CRACK_HALF_LENGTH, SOURCE_OFFSET
    if not all(np.isfinite(v) for v in (b,h,a,d,CRACK_ANGLE_DEG)):
        raise ValueError('几何参数必须为有限数。')
    if min(b,h,a,d) <= 0 or abs(CRACK_ANGLE_DEG) >= 90:
        raise ValueError('尺寸及偏移距离必须为正，倾角必须在(-90,90)内。')
    theta = np.deg2rad(CRACK_ANGLE_DEG)
    m = np.tan(theta)
    if abs(m)*b >= h:
        raise ValueError('倾斜界面必须与板的左右边相交。')
    tip = a*np.array([np.cos(theta),np.sin(theta)])
    if abs(tip[0]) >= b or abs(tip[1]) >= h:
        raise ValueError('两个裂尖必须位于板内。')
    # Ω1 = { -b <= x <= b, m*x <= y <= h }.
    physical = np.array([[-b,-m*b],[b,m*b],[b,h],[-b,h]])
    # Outward parallel offset: x=±(b+d), y=h+d, y=m*x-d/cosθ.
    shift = d/np.cos(theta)
    outer = np.array([[-b-d,m*(-b-d)-shift],
                      [b+d,m*(b+d)-shift], [b+d,h+d],[-b-d,h+d]])
    return physical, outer, np.stack([-tip,tip]), theta


def plot_mixed_mode_sources(output_dir=None, show=True, save=True):
    physical, outer, tips, theta = geometry()
    src_edges = edge_points(outer,N_SOURCES_PER_EDGE)
    bnd_edges = edge_points(physical,N_BOUNDARY_PER_EDGE)
    sources = np.vstack([p if SHOW_ALL_POINTS else display_subset(p,SHOW_SOURCES_PER_EDGE)
                         for p in src_edges])
    boundary = np.vstack([p if SHOW_ALL_POINTS else display_subset(p,SHOW_BOUNDARY_PER_EDGE)
                          for p in bnd_edges])
    b, h = HALF_WIDTH, HALF_HEIGHT
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.6,7.8))
        fig.subplots_adjust(left=.07,right=.93,bottom=.055,top=.94)
        ax.add_patch(Rectangle((-b,-h),2*b,2*h,fill=False,ec=BLACK,lw=1.15,zorder=1))
        ax.add_patch(Polygon(physical,closed=True,fc='#EEF2E8',ec='none',zorder=0))
        # Physical external edges bounding Ω1: right, top and left.
        ax.plot(physical[[1,2,3,0],0],physical[[1,2,3,0],1],color=BLACK,lw=1.6,zorder=2)
        ax.plot(physical[:2,0],physical[:2,1],color=BLACK,lw=1.05,ls=(0,(4,3)),zorder=2)
        closed=np.vstack([outer,outer[0]])
        ax.plot(closed[:,0],closed[:,1],color=GRAY,lw=1.,ls=(0,(5,4)),zorder=2)
        ax.scatter(sources[:,0],sources[:,1],s=24,c=RED,marker='s',
                   edgecolors='white',linewidths=.4,zorder=4)
        if SHOW_BOUNDARY_POINTS:
            ax.scatter(boundary[:,0],boundary[:,1],s=16,c=BLUE,marker='o',
                       edgecolors='white',linewidths=.4,zorder=4)
        ax.plot(tips[:,0],tips[:,1],color=RED,lw=2.6,zorder=5)
        ax.scatter(tips[:,0],tips[:,1],s=85,marker='*',c=BLACK,zorder=7)
        ax.text(0, .62*h, r'$\Omega_1$',ha='center',va='center',fontsize=22,color=BLACK)
        if SHOW_ANGLE and abs(CRACK_ANGLE_DEG)>1e-8:
            radius=.65*CRACK_HALF_LENGTH
            ax.plot([0,radius*1.18],[0,0],color=GRAY,lw=.8,ls=':',zorder=6)
            t1,t2=sorted([0,CRACK_ANGLE_DEG])
            ax.add_patch(Arc((0,0),2*radius,2*radius,theta1=t1,theta2=t2,
                             color=BLACK,lw=.8,zorder=6))
            middle=theta/2
            ax.text(0.72*radius*np.cos(middle),0.72*radius*np.sin(middle),
                    r'$\theta$',fontsize=13,ha='center',va='center',zorder=7)
        pad=.16*b
        ax.set_xlim(outer[:,0].min()-pad,outer[:,0].max()+pad)
        ax.set_ylim(-h-pad,outer[:,1].max()+pad)
        ax.set_aspect('equal');ax.set_axis_off()
        if SHOW_TITLE:
            ax.set_title(rf'Inclined center crack, $\theta={CRACK_ANGLE_DEG:g}^\circ$',pad=12)
        handles=[Line2D([],[],color=BLACK,lw=1.5,label='Physical boundary'),
                 Line2D([],[],color=BLACK,lw=1.,ls=(0,(4,3)),label='Subdomain interface'),
                 Line2D([],[],color=GRAY,lw=1.,ls=(0,(5,4)),label='Artificial source boundary'),
                 Line2D([],[],color=RED,marker='s',ls='none',markersize=5,label=r'Sources outside $\Omega_1$'),
                 Line2D([],[],color=RED,lw=2.4,label='Crack'),
                 Line2D([],[],color=BLACK,marker='*',ls='none',markersize=9,label='Crack tips')]
        if SHOW_BOUNDARY_POINTS:
            handles.append(Line2D([],[],color=BLUE,marker='o',ls='none',markersize=5,
                                  label='Boundary/interface points'))
        ax.legend(handles=handles, loc='center',
                  bbox_to_anchor=(0.0, -0.68*h), bbox_transform=ax.transData,
                  ncol=1, frameon=False, fontsize=13, handlelength=2.1,
                  labelspacing=.60, borderaxespad=0)
        paths=[]
        if save:
            if output_dir is None:
                base=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
                output_dir=base/'mixed_mode_source_figures'
            output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
            for suffix in ('png','pdf','svg'):
                path=output_dir/f'mixed_mode_sources_theta_{CRACK_ANGLE_DEG:g}.{suffix}'
                fig.savefig(path,dpi=400,bbox_inches='tight',pad_inches=.12,facecolor='white')
                paths.append(path)
        if show:plt.show()
        plt.close(fig)
    print(f'示意源点设置：{4*N_SOURCES_PER_EDGE}；显示：{len(sources)}。')
    print('源点位于Ω1外；界面下方源点可以位于完整板的内部。')
    for path in paths:print(f'已保存：{path.resolve()}')
    return paths


if __name__=='__main__':
    plot_mixed_mode_sources(OUTPUT_DIR,show=SHOW_FIGURE,save=SAVE_FIGURES)
