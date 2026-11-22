"""Show requested FOV36/651 and FOV124/1197 cells with annotated HEV neighbors."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.marker_ownership import meca79_motif_pairs as pairs

OUT=pairs.OUT/'requested_cells'
REQUESTS=[('FOV36',651),('FOV124',1197)]
TITLES=['Pair context','MECA-79: pair','MECA-79: HEV','CD31: HEV','SMA: HEV','DNA: HEV']


def draw(ax,case,j):
    if j!=4:
        pairs.draw(ax,case,j if j<4 else 4)
        return
    sl=case['hev_region']
    pixels=pairs.imaging.normalized(case['images']['SMA'],case['limits']['SMA'],sl)
    ax.imshow(pixels[:,:,None]*np.array((0,1,0)),interpolation='nearest')
    pairs.outlines(ax,case['mask'],sl,case['u'],case['h'])
    pairs.imaging.scale_bar(ax,case['mask'][sl].shape,10)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)


def figure(cases):
    fig,axes=plt.subplots(len(cases),6,figsize=(12,len(cases)*2.5+1),squeeze=False)
    for i,case in enumerate(cases):
        for j,ax in enumerate(axes[i]):
            draw(ax,case,j)
            if i==0:ax.set_title(TITLES[j],fontsize=9)
    fig.subplots_adjust(left=.16,right=.99,top=.85,bottom=.13,wspace=.08,hspace=.25)
    for i,case in enumerate(cases):
        pos=axes[i,0].get_position()
        fig.text(.075,(pos.y0+pos.y1)/2,
            f'P{case["patient"]} · {case["fov"]}\nU {case["u"]}: {case["u_mean"]:.3f}\nHEV {case["h"]}: {case["h_mean"]:.3f}',
            ha='center',va='center',fontsize=8)
    fig.suptitle('Requested motif-I unidentified cells and annotated HEV neighbors',fontsize=13,y=.97)
    fig.text(.53,.075,'Dashed white: unidentified mask  |  Solid cyan: annotated HEV mask  |  Values: whole-mask mean MECA-79',ha='center',fontsize=8)
    fig.text(.53,.035,'Merge: MECA-79 green / DNA blue. Separate marker channels green; DNA grayscale. Bars: 20 px (pair), 10 px (HEV).',ha='center',fontsize=7)
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(pairs.OUT/'all_unidentified_HEV_pairs.csv')
    spatial=pd.read_csv(pairs.imaging.DATA/'mapped_cell_types.csv')
    cases=[];records=[]
    for fov,uid in REQUESTS:
        matches=table[table.FOV.eq(fov)&table.unidentified_cell_id.eq(uid)]
        assert len(matches)==1
        row=matches.iloc[0]; u,h=int(uid),int(row.hev_cell_id)
        with ZipFile(row.archive) as archive:
            mask=pairs.imaging.read_image(archive,row.mask_member)
            images={m:pairs.imaging.read_image(archive,f'TIFs/{m}.tif') for m in ['MECA-79','CD31','SMA','dsDNA']}
        assert all(im.shape==mask.shape for im in images.values())
        frame=spatial[spatial.fov.eq(fov)&spatial['patient number'].eq(row.patient)].set_index('cell_id')
        for cid,expected in [(u,'unidentified'),(h,'hevs')]:
            assert frame.loc[cid].pred_mapped==expected
            y,x=np.where(mask==cid);cell=frame.loc[cid]
            assert len(y) and np.hypot(y.mean()-cell['centroid-0'],x.mean()-cell['centroid-1'])<.75
        um=float(images['MECA-79'][mask==u].mean());hm=float(images['MECA-79'][mask==h].mean())
        assert np.isclose(um,row.unidentified_mean_MECA79) and np.isclose(hm,row.hev_mean_MECA79)
        limits={m:max(float(np.quantile(im[im>0],.995)),1) if (im>0).any() else 1 for m,im in images.items()}
        cases.append(dict(fov=fov,patient=int(row.patient),u=u,h=h,mask=mask,images=images,limits=limits,
                          u_mean=um,h_mean=hm,pair_region=pairs.region(mask,[u,h],18),hev_region=pairs.region(mask,[h],12)))
        record=row.to_dict();record['display_limits_json']=json.dumps(limits);records.append(record)
        fig=figure([cases[-1]])
        for ext in ['png','pdf']:fig.savefig(OUT/f'{fov}_cell_{u}.{ext}',dpi=300)
        plt.close(fig)
    fig=figure(cases)
    for ext in ['png','pdf']:fig.savefig(OUT/f'requested_cells.{ext}',dpi=300)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'manifest.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requests=REQUESTS,display='whole-FOV positive-pixel 99.5th percentile; linear; no smoothing or background subtraction',
        annotations='existing mapped labels; not independently validated'),indent=2))
    print(f'Complete: {OUT}',flush=True)


if __name__=='__main__':main()
