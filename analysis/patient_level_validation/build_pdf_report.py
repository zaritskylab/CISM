"""Publish the executed analysis as a PDF, including every statistical family.

Uses installed Matplotlib only for PDF typesetting. No analyses are rerun.
The numerical appendices retain every nonredundant field from result CSVs.
"""
from pathlib import Path
import json
import math
import os
import re
import textwrap

os.environ.setdefault('MPLCONFIGDIR','/tmp/cism_patient_validation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'CISM_patient_level_validation_report.pdf'
BLUE='#244b68'
INK='#263441'
SHORT={0:'B',1:'CD4 APC',2:'CD4 T',3:'Treg',4:'CD8 T',5:'DC',6:'GC B',7:'HEV',8:'MAC',9:'Mem CD4',10:'NK',11:'Neut',12:'Stroma',14:'Unid',15:'Vessel'}

def plain(s):
    s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1',s)
    return s.replace('**','').replace('`','')

def number(v):
    if pd.isna(v):return '—'
    if isinstance(v,(bool,np.bool_)):return 'yes' if v else 'no'
    if isinstance(v,(float,np.floating)):
        return f'{v:.6g}'
    return str(v)

class Report:
    def __init__(self,pdf):
        self.pdf=pdf;self.fig=None;self.y=0;self.pages=[];self.heading='Patient-level validation'

    def footer(self,fig,title):
        fig.text(.055,.025,'CISM • Patient-level statistical validation • 9 November 2026',fontsize=8,color='#637383')
        fig.text(.945,.025,str(len(self.pages)+1),ha='right',fontsize=8,color='#637383')

    def close(self):
        if self.fig is not None:
            self.footer(self.fig,self.heading);self.pdf.savefig(self.fig)
            self.pages.append(self.heading)
            if len(self.pages) in [1,2]:self.fig.savefig(BASE/f'pdf_preview_page_{len(self.pages):02d}.png',dpi=110)
            plt.close(self.fig);self.fig=None

    def new(self,title=None,size=(8.27,11.69)):
        self.close();self.heading=title or self.heading
        self.fig=plt.figure(figsize=size,facecolor='white')
        self.fig.text(.065,.965,self.heading,fontsize=9,color=BLUE,weight='bold',va='top')
        self.y=.925

    def text(self,s,size=10.5,bold=False,gap=.011):
        s=plain(s)
        if not s:return
        width=96 if size<=11 else int(96*10.5/size)
        lines=textwrap.wrap(s,width=width,break_long_words=True,break_on_hyphens=False) or ['']
        step=size*1.4/(11.69*72)
        while lines:
            if self.fig is None:self.new()
            available=int((self.y-.065)/step)
            if available<2:self.new();continue
            portion=lines[:available];lines=lines[available:]
            self.fig.text(.065,self.y,'\n'.join(portion),fontsize=size,color=INK,va='top',linespacing=1.4,weight='bold' if bold else 'normal')
            self.y-=len(portion)*step+gap

    def title(self,s,level=2):
        if level==1:
            self.new('Analysis report');self.text(s,20,True,.03);return
        if self.fig is None or self.y<.23:self.new(s)
        self.y-=.012;self.text(s,15 if level==2 else 12,True,.017)

    def figure(self,path,title,caption=''):
        self.close()
        im=Image.open(path);ratio=im.width/im.height
        size=(11.69,8.27) if ratio>1.45 else (8.27,11.69)
        fig=plt.figure(figsize=size,facecolor='white')
        fig.text(.05,.955,title,fontsize=14,color=BLUE,weight='bold',va='top')
        box=(.04,.11,.92,.76)
        ax=fig.add_axes(box);ax.imshow(im);ax.axis('off')
        if caption:fig.text(.055,.08,'\n'.join(textwrap.wrap(caption,145 if ratio>1.45 else 103)),fontsize=8,color=INK,va='top')
        self.footer(fig,title);self.pdf.savefig(fig);self.pages.append(title)
        if 'Complete spatial' in title:fig.savefig(BASE/'pdf_preview_spatial_heatmaps.png',dpi=100)
        plt.close(fig)

    def tables(self,data,title,subtitle='',rows=34,widths=None):
        self.close()
        if widths is None:
            if len(data.columns)==2:widths=[.35,.65]
            elif len(data.columns)>6:widths=[.23]+[.77/(len(data.columns)-1)]*(len(data.columns)-1)
            else:widths=[1/len(data.columns)]*len(data.columns)
        # A3 landscape leaves 20+ numerical columns readable at normal 8pt type.
        for start in range(0,len(data),rows):
            part=data.iloc[start:start+rows]
            fig=plt.figure(figsize=(16.54,11.69),facecolor='white')
            fig.text(.04,.955,title,fontsize=16,color=BLUE,weight='bold',va='top')
            fig.text(.04,.914,f'Rows {start+1}–{start+len(part)} of {len(data)}. '+subtitle,fontsize=9,color=INK,va='top')
            ax=fig.add_axes([.035,.07,.93,.80]);ax.axis('off')
            capacities=[max(5,int(w*16.54*72*.93/4.3)) for w in widths]
            cells=[[textwrap.fill(number(v),width=capacities[k],break_long_words=True,break_on_hyphens=False) for k,v in enumerate(row)] for row in part.itertuples(index=False,name=None)]
            headers=[textwrap.fill(str(c).replace('_',' '),width=capacities[k],break_long_words=True) for k,c in enumerate(data.columns)]
            t=ax.table(cellText=cells,colLabels=headers,loc='upper center',cellLoc='right',colWidths=widths)
            t.auto_set_font_size(False);t.set_fontsize(8 if len(data.columns)<=20 else 7)
            t.scale(1,1.4)
            for (r,c),cell in t.get_celld().items():
                line_count=max(v.count('\n')+1 for v in (headers if r==0 else cells[r-1]))
                cell.set_height(max(.022,line_count*.019))
                cell.set_edgecolor('#d8e0e7');cell.set_linewidth(.4)
                if r==0:cell.set_facecolor(BLUE);cell.set_text_props(color='white',weight='bold',fontsize=7.5)
                elif r%2:cell.set_facecolor('#f0f5f8')
                if c==0:cell.set_text_props(ha='left')
            self.footer(fig,title);self.pdf.savefig(fig);self.pages.append(title)
            if title.startswith('Appendix B') and start==0:fig.savefig(BASE/'pdf_preview_result_table.png',dpi=100)
            plt.close(fig)

def additional_plots():
    d=BASE/'interactions'
    sp=pd.read_csv(d/'stats_spatial_enrichment.csv');between=pd.read_csv(d/'stats_group_proportions.csv')
    codes=list(SHORT);names=list(SHORT.values());lookup={c:i for i,c in enumerate(codes)}
    fig,axs=plt.subplots(1,2,figsize=(15,7))
    for ax,group in zip(axs,['NN','NP']):
        a=np.full((15,15),np.nan);marks=[]
        for row in sp[sp.group==group].itertuples():
            i,j=lookup[row.code_a],lookup[row.code_b];a[i,j]=np.log2(row.mean_enrichment)
            if row.p_upper_holm<.05:marks.append((i,j))
        im=ax.imshow(a,cmap='RdBu_r',vmin=-3,vmax=3)
        for i,j in marks:ax.text(j,i,'*',ha='center',va='center',fontsize=10,color='black')
        ax.set_xticks(range(15),names,rotation=70,ha='right');ax.set_yticks(range(15),names);ax.set_title(f'{group}: log₂ mean patient enrichment',fontsize=13)
    fig.suptitle('All 120 eligible interaction types per group\n* Holm-adjusted upper-tail p < 0.05 across 240 tests',fontsize=14)
    fig.subplots_adjust(left=.07,right=.86,bottom=.19,top=.84,wspace=.35)
    color_axis=fig.add_axes([.91,.25,.014,.5])
    fig.colorbar(im,cax=color_axis,label='log₂ observed / spatial-null expected')
    fig.savefig(d/'all_interactions_spatial_heatmaps.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,6.5))
    arrays=[]
    for col in ['rank_biserial_NN_minus_NP','p_holm']:
        a=np.full((15,15),np.nan)
        for row in between.itertuples():a[lookup[row.code_a],lookup[row.code_b]]=getattr(row,col)
        arrays.append(a)
    im=axs[0].imshow(arrays[0],cmap='RdBu_r',vmin=-1,vmax=1);fig.colorbar(im,ax=axs[0],shrink=.7);axs[0].set_title('Rank-biserial effect: NN minus NP')
    im=axs[1].imshow(arrays[1],cmap='viridis',vmin=0,vmax=1);fig.colorbar(im,ax=axs[1],shrink=.7);axs[1].set_title('Holm-adjusted p: all values ≥ 0.897')
    for ax in axs:ax.set_xticks(range(15),names,rotation=70,ha='right');ax.set_yticks(range(15),names)
    fig.suptitle('NN versus NP patient interaction proportions: complete 120-pair results',fontsize=14);fig.tight_layout();fig.savefig(d/'all_interactions_between_group_heatmaps.png',dpi=180);plt.close(fig)

def appendices(r):
    r.new('Numerical appendices')
    r.title('Complete numerical results',1)
    r.text('The appendices include every cell type and every interaction in every executed statistical family, including sensitivity analyses. Values use six significant digits. Complete unrounded machine-readable results and patient-level inputs remain in the companion analysis directory. Input cells, FOVs and edges are not independent statistical replicates.')
    r.text('Compact interaction names: B = B cell; CD4 APC = CD4 antigen-presenting cell; CD4 T = CD4 T cell; Treg = regulatory CD4 T cell; CD8 T = CD8 T cell; DC = dendritic cell; GC B = germinal-center B cell; HEV = high endothelial venule; MAC = macrophage; Mem CD4 = memory CD4 T cell; NK = natural killer cell; Neut = neutrophil; Stroma = stroma; Unid = unidentified; Vessel = blood vessel.')
    r.text('Table conventions: NN and NP n are independent patient sample sizes. IQR denotes Q25–Q75 endpoints where presented as a range, and IQR width where explicitly labeled. Delta and rank-biserial effects are NN minus NP. Holm families are stated in each title. Spatial bootstrap intervals describe uncertainty of the equal-patient mean ratio; null intervals describe the distribution under shuffled labels. Rank is descriptive among sufficiently supported types. An em dash means not supported for ranking, not a missing test.')
    for variant in ['primary_original_graph_retained_cells','sensitivity_all_raw_cells','sensitivity_tutorial_graph_nodes','sensitivity_without_tumor']:
        folder=BASE/'composition'/variant;df=pd.read_csv(folder/'rank_sum_tests.csv');glob=json.loads((folder/'global_jensen_shannon_test.json').read_text())
        out=pd.DataFrame({'Cell type':df.cell_type,'n NN':df.n_NN,'n NP':df.n_NP,'NN Q25':df.NN_q25,'NN med':df.NN_median,'NN Q75':df.NN_q75,'NN IQR':df.NN_IQR,'NP Q25':df.NP_q25,'NP med':df.NP_median,'NP Q75':df.NP_q75,'NP IQR':df.NP_IQR,'Delta pp':df.median_difference_NN_minus_NP,'U NN':df.U_NN,'Rank bis.':df.rank_biserial_NN_minus_NP,'Raw p':df.p_raw,'Holm p':df.p_holm,'Zero NN':df.n_zero_NN,'Zero NP':df.n_zero_NP,'Distinct':df.n_distinct_pooled})
        widths=[.13]+[.87/(len(out.columns)-1)]*(len(out.columns)-1)
        r.tables(out,'Appendix A • Composition • '+variant.replace('_',' '),f"Holm family = {len(df)} types. All composition values in %. Global JSD {glob['distance_base2']:.9g}; permutation p {glob['p_empirical']:.9g}; B={glob['permutations']}.",widths=widths)
    for fname,label in [('stats_group_proportions.csv','Primary edge proportions'),('stats_group_enrichment.csv','Exploratory patient enrichment-score comparison'),('stats_equal_fov_weight_sensitivity.csv','Equal-FOV-within-patient proportions'),('stats_historical_matrix_sensitivity.csv','Historical doubled-diagonal matrix entries')]:
        df=pd.read_csv(BASE/'interactions'/fname)
        # Percentages for proportions/matrix entries; ratios retain natural units.
        factor=1 if 'group_enrichment' in fname else 100
        out=pd.DataFrame({'Interaction':[SHORT[a]+' / '+SHORT[b] for a,b in zip(df.code_a,df.code_b)],'n NN':df.n_NN,'n NP':df.n_NP,'NN Q25':df.q25_NN*factor,'NN med':df.median_NN*factor,'NN Q75':df.q75_NN*factor,'NP Q25':df.q25_NP*factor,'NP med':df.median_NP*factor,'NP Q75':df.q75_NP*factor,'Delta':df.median_difference_NN_minus_NP*factor,'U NN':df.U_NN,'Rank bis.':df.rank_biserial_NN_minus_NP,'Raw p':df.p_raw,'Holm p':df.p_holm})
        widths=[.19]+[.81/(len(out.columns)-1)]*(len(out.columns)-1)
        r.tables(out,'Appendix B • '+label,'Holm family = 120 interactions. '+('Values are enrichment ratios.' if factor==1 else 'Quartiles and differences expressed as percentages / percentage points.'),widths=widths)
    for fname,label in [('stats_spatial_enrichment.csv','Primary FOV-wide null'),('stats_degree_stratified.csv','Degree-stratified null sensitivity'),('stats_pseudocount_0.1.csv','Pseudocount 0.1 sensitivity'),('stats_pseudocount_1.csv','Pseudocount 1 sensitivity')]:
        df=pd.read_csv(BASE/'interactions'/fname)
        out=pd.DataFrame({'Group / interaction':[g+' '+SHORT[a]+'/'+SHORT[b] for g,a,b in zip(df.group,df.code_a,df.code_b)],'n':df.n_patients,'Mean E':df.mean_enrichment,'Median E':df.median_enrichment,'CI low':df.bootstrap95_low,'CI high':df.bootstrap95_high,'Obs sum':df.observed_sum,'Exp sum':df.expected_sum,'n exp≥5':df.n_patients_expected_ge5,'Rank':df.rank_supported_mean_enrichment,'Null low':df.null_stat95_low,'Null high':df.null_stat95_high,'Exceed':df.exceedances_upper,'Upper p':df.p_upper,'Upper Holm':df.p_upper_holm,'2-sided p':df.p_two_sided,'2-sided Holm':df.p_two_sided_holm})
        widths=[.17]+[.83/(len(out.columns)-1)]*(len(out.columns)-1)
        r.tables(out,'Appendix C • Spatial enrichment • '+label,f"Holm family = 240 (120 pairs × 2 groups). Permutations/FOV = {df.permutations.iloc[0]}; pseudocount = {df.pseudocount.iloc[0]:g}. CI = patient-bootstrap 95%; null = group-statistic 95% range.",widths=widths)
    for name,label in [('candidate_leave_one_patient_out_descriptive.csv','Leave-one-patient-out group mean ratios'),('candidate_fov_heterogeneity_descriptive.csv','Within-patient FOV heterogeneity (descriptive)')]:
        df=pd.read_csv(BASE/'interactions'/name)
        df['Interaction']=[SHORT[a]+'/'+SHORT[b] for a,b in zip(df.code_a,df.code_b)]
        df=df.drop(columns=['code_a','code_b','interaction'],errors='ignore')
        # Shorten column labels for readable appendix widths.
        df=df.rename(columns={'patient_id':'Patient','group':'Group','mean_enrichment':'Mean ratio','leave_one_patient_out_mean_min':'LOO mean min','leave_one_patient_out_mean_max':'LOO mean max','fov_proportion_min':'FOV prop min','fov_proportion_max':'FOV prop max','fov_enrichment_min':'FOV ratio min','fov_enrichment_max':'FOV ratio max','fovs':'FOVs'})
        r.tables(df,'Appendix D • '+label,'Descriptive only; no FOV-level independent inference. FOV proportions in natural 0–1 units.')

def markdown(r,path,used):
    lines=path.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('!['):
            match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
            if match:
                target=(path.parent/match[2]).resolve();r.figure(target,match[1]);used.add(target)
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rows.append([plain(v.strip()) for v in lines[i].strip().strip('|').split('|')]);i+=1
            if len(rows)>2:
                df=pd.DataFrame(rows[2:],columns=rows[0]);r.tables(df,'Report summary table',rows=22)
            continue
        if line.startswith('#'):
            m=re.match(r'(#+)\s+(.*)',line);r.title(m[2],len(m[1]));i+=1;continue
        # Join prose paragraphs, retaining list items as their own paragraphs.
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|- |\d+\. )',lines[i].strip()):paragraph.append(lines[i].strip());i+=1
        r.text(' '.join(paragraph))

def main():
    additional_plots();used=set()
    with PdfPages(OUT,metadata={'Title':'CISM: Patient-level statistical validation of Fig. 2B and Fig. S2','Author':'CISM analysis workspace','Subject':'Complete statistical results, plots, provenance and independent validation','Keywords':'melanoma NN NP patient-level Wilcoxon spatial enrichment'}) as pdf:
        r=Report(pdf)
        markdown(r,BASE/'REPORT.md',used)
        r.new('Additional plots')
        r.title('Additional plots and full interaction maps',1)
        r.text('The next pages include complete interaction maps and the remaining sensitivity plots. All plots derive from saved patient-level results. They introduce no new tests or result-selected interaction targets.')
        for name,title in [('all_interactions_spatial_heatmaps.png','Complete spatial enrichment maps'),('all_interactions_between_group_heatmaps.png','Complete NN versus NP interaction results')]:
            r.figure(BASE/'interactions'/name,title,'Abbreviations are defined in the numerical appendix. Upper triangles contain unique unordered interactions; blank lower triangles are symmetric duplicates.');used.add((BASE/'interactions'/name).resolve())
        for path in sorted(BASE.rglob('*.png')):
            if path.resolve() in used or path.name.startswith('pdf_preview'):continue
            title=path.stem.replace('_',' ').capitalize()
            if path.name=='patient_composition.png':title='Composition sensitivity: '+path.parent.name.replace('_',' ')
            r.figure(path,title,'Source: '+str(path.relative_to(BASE)))
        appendices(r)
        r.new('Independent validation and provenance')
        r.title('Independent validation: complete review',1);markdown(r,BASE/'validation/review.md',used)
        r.new('Data provenance audit')
        r.title('Data provenance: full audit',1);markdown(r,BASE/'provenance_audit.md',used)
        r.close()
        manifest={'pdf':str(OUT),'pages':len(r.pages),'page_titles':r.pages,'statistical_tables':{'composition_variants':4,'between_group_families':4,'spatial_families':4},'plots_included':len(list(BASE.rglob('*.png')))-len(list(BASE.glob('pdf_preview*.png')))}
        (BASE/'pdf_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'pdf':str(OUT),'pages':manifest['pages'],'bytes':OUT.stat().st_size}))

if __name__=='__main__':main()
