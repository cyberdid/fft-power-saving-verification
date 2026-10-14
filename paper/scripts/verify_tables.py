"""Validate publication tables against the selected data, with missing-data failures.

--mode archive/full checks the complete published scope. --mode quick checks the
explicit quick-run subset and reports omitted experiments. No archive data is substituted
for a fresh run. Numerical comparisons use log error because tiny mp errors underflow float.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SUPERS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')


def number(text):
    text=text.strip().replace(' ','').replace('\u202f','').replace(',','.').replace('−','-')
    if '·10' in text:
        mantissa, exponent=text.split('·10',1)
        return Decimal(mantissa)*Decimal(10)**int(exponent.translate(SUPERS).strip('^{}'))
    if text.startswith('2') and len(text)>1 and text[1] in '⁰¹²³⁴⁵⁶⁷⁸⁹':
        return Decimal(2)**int(text[1:].translate(SUPERS))
    return Decimal(text)


def manuscript_tables(path):
    result={};lines=Path(path).read_text().splitlines()
    for i,line in enumerate(lines):
        match=re.match(r'^Таблиця (\d+[a-zа-я]?)\.',line)
        if not match:continue
        rows=[];started=False
        for following in lines[i+1:]:
            if following.startswith('|'):
                cells=[c.strip() for c in following.strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                started=True
            elif started or following.startswith('Таблиця '):break
        if rows:
            if any(len(r)!=len(rows[0]) for r in rows):raise ValueError(f'malformed table {match[1]}')
            result[match[1]]=rows[1:]
    return result


class Validation:
    def __init__(self,data,article,mode):
        self.data=Path(data);self.mode=mode;self.article=manuscript_tables(article)
        self.failures=[];self.checked=0;self.loaded={};self.omitted=[]

    def require(self,label,condition):
        self.checked+=1
        if not condition:self.failures.append(label)

    def rows(self,name,required=True):
        if name in self.loaded:return self.loaded[name]
        path=self.data/name
        if not path.is_file():
            if required:self.failures.append(f'missing file: {name}')
            self.loaded[name]=[];return []
        with path.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        if required and not rr:self.failures.append(f'empty file: {name}')
        self.loaded[name]=rr;return rr

    def index(self,name,keys,required=True):
        out={}
        for row in self.rows(name,required):
            try:key=tuple(row[k] for k in keys)
            except KeyError:
                self.failures.append(f'{name}: invalid schema (required {keys})');continue
            self.require(f'{name}: duplicate key {key}',key not in out)
            out[key]=row
        return out

    def equal(self,label,actual,expected):
        try:ok=number(str(actual))==number(str(expected))
        except (InvalidOperation,ValueError,TypeError):ok=False
        self.require(label,ok)

    def near(self,label,actual,expected,tol):
        try:
            a,b=number(str(actual)),number(str(expected))
            ok=a.is_finite() and b.is_finite() and abs(a-b)<=Decimal(str(tol))
        except (InvalidOperation,ValueError,TypeError):ok=False
        self.require(label,ok)

    def log_near(self,label,actual,expected,tol=0.5):
        try:
            a,b=number(str(actual)),number(str(expected))
            ok=a.is_finite() and b.is_finite() and a>0 and b>0 and abs(a.log10()-b.log10())<=Decimal(str(tol))
        except (InvalidOperation,ValueError,TypeError):ok=False
        self.require(label,ok)

    def structural(self):
        st=self.index('strict_tables.csv',('r','rule','input'))
        for row in self.article['5']:
            r=str(int(number(row[0])))
            for rule,column in (('adaptive',2),('chunk1',3)):
                got=st.get((r,rule,'seed'))
                self.require(f'table5 required r={r} {rule}',got is not None)
                if got:
                    self.equal(f'table5 depth r={r} {rule}',got['depth'],row[column])
                    self.equal(f'gate count r={r} {rule}',got['ops'],int(got['shears'])+int(got['scales']))
            got=st.get((r,'adaptive','seed'))
            if got:self.equal(f'table5 chunk r={r}',got['b_adm'],row[1])
        for row in self.article['4']:
            r=str(int(number(row[0])));rule='adaptive' if 'адаптив' in row[1] else 'chunk1'
            got=st.get((r,rule,'seed'))
            if got:
                for field,column in (('depth',2),('layers',3),('ops',4)):
                    self.equal(f'table4 {field} {r}/{rule}',got[field],row[column])
        for row in self.article['7']:
            r=int(number(row[0]));got=st.get((str(r),'adaptive','seed'))
            self.require(f'table7 required {r}',got is not None)
            if not got:continue
            word=8*int(got['shears'])+6*int(got['scales']);chain=2*word+18*r
            fft=5*r*int(math.log2(r));circulant=20*r*int(math.log2(2*r))+12*r;direct=4*r*(r+1)
            for column,value in ((1,word),(2,chain),(3,fft),(4,circulant),(5,direct)):
                self.equal(f'table7 flops {r}/column{column}',value,row[column])
            for column,value in ((6,chain/fft),(7,word/circulant),(8,word/direct)):
                self.near(f'table7 rounded ratio {r}/column{column}',value,row[column],0.51)
        topo=self.index('v6/topology.csv',('r','rule','input'))
        for row in self.article['5']:
            r=str(int(number(row[0])))
            for rule in ('adaptive','chunk1'):
                group=[topo.get((r,rule,kind)) for kind in ('seed','seed2','ones')]
                self.require(f'topology three inputs {r}/{rule}',all(group))
                if all(group):
                    self.require(f'topology identity {r}/{rule}',len({tuple(g[k] for k in ('depth','layers','shears','scales','hash')) for g in group})==1)
                    canonical=st.get((r,rule,'seed'))
                    if canonical:
                        for k in ('depth','layers','shears','scales'):self.equal(f'topology counts {r}/{rule}/{k}',group[0][k],canonical[k])
        models=self.index('depth_model_strict.csv',('e',))
        for row in self.article['6b']:
            r=int(number(row[0]));e=int(math.log2(r));got=models.get((str(e),))
            self.require(f'model e={e}',got is not None)
            if got:
                self.equal(f'model depth e={e}',got.get('depth',got.get('d')),row[2])
                self.equal(f'model chunk e={e}',got['b'],row[1])
                self.near(f'model ratio e={e}',got['ratio'],row[3],0.06)
        return st

    def conditioning(self):
        import mpmath as mp
        cond=self.index('v6/cond.csv',('r',))
        with mp.workdps(60):
            constant=mp.im(mp.polylog(2,mp.exp(mp.j*mp.pi/3)))/(mp.pi*mp.log(10))
            for row in self.article['2a']:
                r=int(number(row[0]));partial=[mp.mpf(0)]
                for k in range(1,r):partial.append(partial[-1]+mp.log10(2*mp.sin(mp.pi*k/r)))
                finite=float(max(partial)-min(partial))
                self.near(f'finite diagonal condition {r}',finite,row[2],0.006)
                self.near(f'Clausen leading term {r}',float(constant*r),row[3],0.006)
                if row[1]!='—':
                    got=cond.get((str(r),));self.require(f'condition data {r}',got is not None)
                    if got:
                        self.near(f'condition T {r}',got['log10_condT'],row[1],0.011)
                        self.near(f'condition diagonal {r}',got['log10_condDh'],row[2],0.011)

    def precision(self):
        mp=self.index('v6/mp_dft.csv',('r','dps','vectors','n'))
        cs=self.index('v6/cslots.csv',('r','dps','vectors','n'))
        for label,table,points in (('1b',mp,self.article['1b']),('1c',cs,self.article['1c'])):
            for row in points:
                r=int(number(row[0]));kind='basis' if 'базис' in row[1] else 'random'
                n=r if kind=='basis' else int(row[1].split()[0])
                dps_values=row[2].split('/');error_values=row[3 if label=='1b' else 4].split('/')
                losses=row[4 if label=='1b' else 5].split('/')
                self.require(f'table{label} paired precisions r={r}',len(dps_values)==len(error_values)==len(losses))
                for dps,expected_error,expected_loss in zip(dps_values,error_values,losses):
                    dps=int(number(dps));key=(str(r),str(dps),kind,str(n))
                    if self.mode=='quick' and label=='1c' and r==256:
                        self.omitted.append(f'table1c {key}');continue
                    got=table.get(key);self.require(f'table{label} required {key}',got is not None)
                    if got:
                        self.log_near(f'table{label} error {key}',got['max_rel_err'],expected_error)
                        self.near(f'table{label} lost {key}',got['digits_lost'],expected_loss,0.5)
                        if label=='1b':
                            self.near(f'H growth {key}',got['log10_maxH'],row[5],0.11)
                            self.near(f'inverse H growth {key}',got['log10_maxinvH'],row[6],0.11)
                        if label=='1c':
                            self.equal(f'C-slot blocks {key}',got['blocks_per_vector'],row[3])
                            self.require(f'C-slot conjugate DAG {key}',got['conj']=='conj-DAG')
        for name,points in (('mp',mp),('cslots',cs)):
            for key,row in points.items():
                try:
                    error=Decimal(row['max_rel_err']);lost=Decimal(row['digits_lost']);expected=Decimal(row['dps'])+error.log10()
                    ok=error.is_finite() and error>0 and lost.is_finite() and abs(lost-expected)<Decimal('0.11')
                except (InvalidOperation,ValueError):ok=False
                self.require(f'{name} finite/consistent error {key}',ok)
        floats={}
        for row in self.rows('dft_precision.csv'):
            if row['kind']=='f64_dft':
                self.require(f'duplicate float64 r={row["r"]}',row['r'] not in floats);floats[row['r']]=row
        for row in self.article['2']:
            key=str(int(number(row[0])));got=floats.get(key)
            self.require(f'float64 required {key}',got is not None)
            if got:
                self.log_near(f'float64 error {key}',got['max_rel_err'],row[2],0.1)
                if self.mode!='archive':
                    self.log_near(f'float64 median {key}',got.get('median'),row[3],0.1)
        toe=self.index('v6/toeplitz.csv',('r','dps','vectors'))
        for row in self.article['3']:
            r=int(number(row[0]))
            for dps,error,lost in zip(row[1].split('/'),row[2].split('/'),row[3].split('/')):
                key=(str(r),str(int(number(dps))),'3')
                if self.mode=='quick' and r>352:self.omitted.append(f'table3 {key}');continue
                got=toe.get(key);self.require(f'Toeplitz required {key}',got is not None)
                if got:
                    self.log_near(f'Toeplitz error {key}',got['max_rel_err'],error)
                    self.near(f'Toeplitz lost {key}',got['digits_lost'],lost,0.5)
                    if self.mode!='archive':self.near(f'Toeplitz reciprocal growth {key}',got.get('log10_maxg'),row[4],0.11)

    def exact(self):
        if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
        from allength.exact_field import _is_prime
        rr=self.rows('v6/exact.csv')
        def totient(n):
            result=n;factor=2
            while factor*factor<=n:
                if n%factor==0:
                    result-=result//factor
                    while n%factor==0:n//=factor
                factor+=1
            if n>1:result-=result//n
            return result
        def prime_id(row):
            match=re.fullmatch(r'F_p p=(\d+) \(p = 1 mod (\d+)\)',row['field'])
            return int(match[1]) if match else None
        for row in rr:
            r=int(row['r']);order=math.lcm(r,1<<(8*r-1).bit_length())
            cyc=re.fullmatch(r'Q\(zeta_(\d+)\) \[degree (\d+)\]',row['field'])
            prime=re.fullmatch(r'F_p p=(\d+) \(p = 1 mod (\d+)\)',row['field'])
            self.require(f'exact field schema r={r}',bool(cyc or prime))
            if cyc:
                self.equal(f'exact field order r={r}',cyc[1],order)
                self.equal(f'exact cyclotomic degree r={r}',cyc[2],totient(order))
            if prime:
                value=int(prime[1])
                self.equal(f'exact field order r={r}',prime[2],order)
                self.require(f'exact prime field r={r}',2**61<value<2**64 and _is_prime(value))
                self.equal(f'exact prime congruence r={r}',value%order,1)
            self.equal(f'exact mismatches {row["r"]}/{row["field"]}/{row["path"]}',row['mismatches'],0)
            self.equal(f'exact entry count {row["r"]}/{row["path"]}',row['entries'],int(row['r'])*int(row['n']))
        for r in (4,7,8,9,12,15,16,32):
            self.require(f'Q(zeta) full matrix r={r}',any(int(x['r'])==r and x['field'].startswith('Q(') and x['path']=='word' and int(x['n'])==r and x['vectors']=='basis' for x in rr))
        for r in (4,8,16):
            self.require(f'Q(zeta) C-slots r={r}',any(int(x['r'])==r and x['field'].startswith('Q(') and x['path']=='C-slots+conj-DAG' and int(x['n'])==r and x['vectors']=='basis' for x in rr))
        for r in (9,15,16,31,32,33,63,64,127,128,255,256,512):
            matching=[x for x in rr if int(x['r'])==r and x['field'].startswith('F_p ') and x['path']=='word']
            required=3 if r<=128 else 2
            self.require(f'Fp distinct primes r={r}',len({prime_id(x) for x in matching if prime_id(x) is not None})>=required)
            n=r if r<=128 else (2 if r==512 else 3)
            self.require(f'Fp vector coverage r={r}',bool(matching) and all(int(x['n'])==n and x['vectors']==('basis' if r<=128 else 'random') for x in matching))
        for r in (16,31,32,33,64,128):
            self.require(f'Fp C-slots r={r}',any(int(x['r'])==r and x['field'].startswith('F_p ') and x['path']=='C-slots+conj-DAG' and int(x['n'])==(2 if r==128 else r) and x['vectors']==('random' if r==128 else 'basis') for x in rr))

    def extended(self,st):
        # Fresh runs must supply data for the stage, growth, intervention, root and timing tables.
        if self.mode=='archive':return
        widths=self.index('depth_model_widths.csv',('r',))
        for row in self.article['6a']:
            r=str(int(number(row[0])));got=widths.get((r,));self.require(f'width model {r}',got is not None)
            if got:
                self.equal(f'width model depth {r}',got['depth'],row[1])
                self.equal(f'width model ASAP {r}',st[(r,'adaptive','seed')]['depth'],row[2])
                self.near(f'width model ratio {r}',int(row[2])/int(got['depth']),row[3].split()[0],0.000006)
        stage=self.index('v6/stagewise.csv',('r','stage'))
        for row in self.article['2b']:
            r=str(int(number(row[0])));phase={'D_H':None,'Tᵀ':'2','Dmid':'3','T':'4'}.get(row[1])
            if phase is None:
                # first/last diagonal occurrence is distinguished by accumulated loss.
                phase='1' if row[3]==row[2] else '5'
            got=stage.get((r,phase));self.require(f'stage r={r}/{phase}',got is not None)
            if got:
                for key,column in (('local_lost',2),('cumul_lost',3),('log10_growth',4),('log10_norm',5)):
                    self.near(f'stage {r}/{phase}/{key}',got[key],row[column],0.6)
        peak=self.index('v6/causal_levels.csv',('r','stage'))
        for row in self.article['2c']:
            key=(str(int(number(row[0]))),'T^T' if row[1]=='Tᵀ' else 'T');got=peak.get(key)
            self.require(f'growth {key}',got is not None)
            if got:
                self.near(f'growth total {key}',got['growth'],row[2],0.11)
                self.near(f'growth phase sum {key}',sum(float(got[k]) for k in ('chunk','direct','leaf')),got['growth'],0.021)
                levels=json.loads(got['levels'])
                self.near(f'growth level sum {key}',sum(levels.values()),got['growth'],0.031)
                for field,column in (('chunk',3),('direct',4)):
                    self.near(f'growth percent {key}/{field}',100*float(got[field])/float(got['growth']),row[column].strip('%'),0.61)
                expected={v:float(number(val)) for v,val in re.findall(r'v(\d+):\s*([^;]+)',row[5])}
                self.require(f'growth levels covered {key}',{v for v,val in levels.items() if val!=0} <= set(expected))
                for v,expected_value in expected.items():self.near(f'growth level {key}/v{v}',levels.get(v,0),expected_value,0.061)
        phase=self.index('v6/phase_precision.csv',('r','stage'))
        for row in self.article['2d']:
            key=(str(int(number(row[0]))),'tword' if row[1]=='Tᵀ' else 'word');got=phase.get(key)
            self.require(f'phase control {key}',got is not None)
            if got:
                self.equal(f'phase DPS {key}',got['dps'],row[2])
                for field,column in (('baseline_lost',3),('chunk_high_lost',4),('direct_high_lost',5)):
                    self.near(f'phase {key}/{field}',got[field],row[column].split('(')[0],0.3)
        roots=self.index('v6/roots.csv',('r',))
        for row in self.article['3a']:
            r=str(int(number(row[0])));got=roots.get((r,));self.require(f'roots {r}',got is not None)
            if got:
                self.near(f'root rho {r}',got['rho'],row[1],0.00011)
                residual=Decimal(got['residual'])
                self.require(f'root residual {r}',residual.is_finite() and 0<=residual<=Decimal('4e-14'))
                self.near(f'root slope {r}',got['slope'],row[3],0.00011)
                inside,degree=row[2].split(' з ')
                self.equal(f'roots inside {r}',got['inside'],inside)
                self.equal(f'root degree {r}',got['degree'],degree)
                self.near(f'root reciprocal slope {r}',got['reciprocal_slope'],row[4],0.00011)
                self.near(f'root predicted growth {r}',got['predicted_growth'],row[5],0.11)
                self.near(f'root observed growth {r}',got['growth'],row[6],0.11)
        timing=self.index('v6/timing.csv',('r',))
        for r in (64,128,256,512):
            got=timing.get((str(r),));self.require(f'timing {r}',got is not None)
            if got:
                self.equal(f'timing gate count {r}',got['gates'],st[(str(r),'adaptive','seed')]['ops'])
                for field in ('build_seconds','apply_seconds','peak_rss_gib','fft_us','dense_us'):
                    try:value=Decimal(got[field]);ok=value.is_finite() and value>0
                    except InvalidOperation:ok=False
                    self.require(f'timing positive {r}/{field}',ok)
        rounds=self.index('v6/rounds.csv',('r','kind'))
        for r in (16,32,48,64):
            got=rounds.get((str(r),'difference'));self.require(f'round equivalence {r}',got is not None and Decimal(got['value'])==0)
        for r in (256,512):
            got=rounds.get((str(r),'conflicts'));self.require(f'round conflicts {r}',got is not None and Decimal(got['value'])==0)

    def run(self):
        st=self.structural();self.conditioning();self.precision();self.exact();self.extended(st)
        return {'mode':self.mode,'checks':self.checked,'failures':self.failures,'omitted':self.omitted,
                'status':'FAIL' if self.failures else 'PASS'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'paper/data')
    parser.add_argument('--article',type=Path,default=ROOT/'paper/article-ua.txt')
    parser.add_argument('--mode',choices=('archive','quick','full'),default='archive')
    parser.add_argument('--output-json',type=Path)
    args=parser.parse_args(argv)
    try:result=Validation(args.data_dir,args.article,args.mode).run()
    except (KeyError,ValueError,InvalidOperation,OSError) as error:
        result={'mode':args.mode,'status':'FAIL','failures':[f'invalid/missing publication data: {error}'],'checks':0,'omitted':[]}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for failure in result['failures']:print('FAIL:',failure)
    for omitted in result['omitted']:print('EXPLICITLY OMITTED:',omitted)
    print(f"VERIFY {result['status']}: {result['checks']} checks, {len(result['failures'])} failures")
    return 1 if result['failures'] else 0


if __name__=='__main__':
    raise SystemExit(main())
