"""数据几何描述的相位不变性、等文件权重和近邻同文件排除核验。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_data_geometry import views, metadata_arrays, nearest_relation


def test_harmonic_magnitude_view_preserves_dc_and_is_invariant_to_cycle_shift():
    """复系数按阶次旋转模拟周期起始位置变化，DC/幅值应不变。"""
    rng = np.random.default_rng(14)
    x = rng.standard_normal((4, 5, 9))
    z = x[..., 1:].reshape(4, 5, 4, 2)
    complex_values = z[..., 0] + 1j * z[..., 1]
    shifted = complex_values * np.exp(1j * np.arange(1, 5) * .73)
    y = x.copy()
    y[..., 1:] = np.stack((shifted.real, shifted.imag), axis=-1).reshape(4, 5, 8)
    a, b = views(x.reshape(4, 45)), views(y.reshape(4, 45))
    assert not np.allclose(a["complex45"], b["complex45"])
    np.testing.assert_allclose(a["dc_amplitude25"], b["dc_amplitude25"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(a["dc_amplitude25"].reshape(4, 5, 5)[..., 0], x[..., 0])


def test_longer_records_do_not_receive_more_distribution_weight():
    """一个文件多切几个相邻周期，不能增加该文件的总统计权重。"""
    metadata = [{"record_id": "a", "label": 0}] * 4 + [{"record_id": "b", "label": 0}]
    labels, ids, weights = metadata_arrays(metadata)
    assert weights[:4].sum() == weights[4] == 1


def test_nearest_neighbors_exclude_all_states_from_same_file():
    """排除同文件全部状态，并能在不同文件找到同类邻居。"""
    metadata = [{"record_id": f"record{copy}", "label": label} for label in range(6) for copy in range(2)]
    matrix = np.repeat(np.arange(6) * 10., 2).reshape(-1, 1)
    result = nearest_relation(matrix, metadata, matrix, metadata, True)
    assert result["equal_class_equal_record_label_agreement_percent"] == 100
    assert result["excluded_same_record"]
