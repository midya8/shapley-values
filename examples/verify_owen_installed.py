"""Run inside a fresh installed environment, outside the source checkout."""
import copy
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import shapleylab
from shapleylab.dividends import decompose_game
from shapleylab.owen_controls import definitions
from shapleylab.owen_report import audit_owen
from shapleylab.owen_report_verify import verify_audit


def main():
    root=Path(shapleylab.__file__).resolve().parent
    assert 'site-packages'in str(root)and not Path.cwd().is_relative_to(root)
    modules=[shapleylab]+[importlib.import_module(m.name)for m in pkgutil.walk_packages(shapleylab.__path__,shapleylab.__name__+'.')]
    origins=[str(Path(m.__file__).resolve())for m in modules];assert len(origins)==20
    assert all(Path(p).is_relative_to(root)for p in origins)
    routes=[[sys.executable,'-m','shapleylab.owen_cli'],[str(Path(sys.executable).parent/'shapleylab-owen')]]
    forged=0
    with tempfile.TemporaryDirectory()as tmp:
        tmp=Path(tmp)
        for control in definitions():
            expected=audit_owen(control['request']);assert verify_audit(expected)
            allocation=expected['allocation']
            decomposition=decompose_game(allocation['coalition_values'], allocation['feature_names'], control['request']['groups'])
            for a,b in zip(decomposition['reconstructed_values'], allocation['coalition_values']):
                assert Fraction(int(a['numerator']),int(a['denominator'])) == Fraction(b)
            for a,b in zip(decomposition['owen_attributions'], allocation['attributions']):
                assert abs(float(Fraction(int(a['numerator']),int(a['denominator'])))-b) <= 1e-12 * max(1,abs(b))
            assert json.loads(json.dumps(decomposition))==decomposition
            req=tmp/'request.json';req.write_text(json.dumps(control['request']))
            for route in routes:
                generated=subprocess.run(route+[str(req)],capture_output=True,check=True);report=json.loads(generated.stdout);assert report==expected
                out=tmp/'report.json';out.write_bytes(generated.stdout)
                verified=subprocess.run(route+[str(out),'--verify'],capture_output=True,check=True);assert json.loads(verified.stdout)['verified']
            for field in ['allocation','replacement_evaluations']:
                bad=copy.deepcopy(expected)
                if field=='allocation':bad[field]['attributions'][0]+=1
                else:bad[field][0]['score']+=1
                out=tmp/'bad.json';out.write_text(json.dumps(bad));result=subprocess.run(routes[0]+[str(out),'--verify'],capture_output=True)
                assert result.returncode==2 and not result.stdout and b'Traceback'not in result.stderr;forged+=1
    assert forged==24
    print(json.dumps({'owen_controls':12,'rational_decompositions':12,'entrypoints':2,'installed_origins':origins,'forged_cli_rejections':forged}))


if __name__=='__main__':main()
