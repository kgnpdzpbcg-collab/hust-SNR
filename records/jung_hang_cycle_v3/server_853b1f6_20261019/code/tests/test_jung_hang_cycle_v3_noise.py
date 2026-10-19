"""检查V3运行器能实际执行两条加噪路径，防止只测模型时漏掉入口错误。"""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from jung_hang_cycle_v3 import noise_records


@pytest.mark.parametrize("validation_snr",[None,20])
def test_noise_record_adapter_executes_training_and_validation(validation_snr):
    """合成完整记录包含断流区；两种入口都应保持该区并达到逐相目标SNR。"""
    time=np.arange(2000)/2000.0
    current=np.stack([np.sin(2*np.pi*10*time+p) for p in range(5)],axis=1)
    current[(time>0.3)&(time<0.7),0]=0
    speed=np.full_like(time,2*np.pi*10)
    raw={"synthetic":(time,current,speed)}
    reference=SimpleNamespace(data=SimpleNamespace(training_noise_schedule=lambda records,seed,epoch:{key:20 for key in records}))
    result=noise_records(reference,raw,42,1,snr=validation_snr)
    noise=result["synthetic"][1]-current
    assert np.all(noise[(time>0.35)&(time<0.65),0]==0)
    snr=10*np.log10(np.mean(current**2,axis=0)/np.mean(noise**2,axis=0))
    np.testing.assert_allclose(snr,np.full(5,20.0),rtol=0,atol=1e-9)
    np.testing.assert_array_equal(result["synthetic"][0],time)
    np.testing.assert_array_equal(result["synthetic"][2],speed)
