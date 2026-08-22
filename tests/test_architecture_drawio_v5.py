import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import generate_architecture_drawio_v5 as architecture


class ArchitectureDrawioV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        architecture.build_spec()

    def test_required_implementation_semantics_are_present(self) -> None:
        text = "\n".join(
            [item.text for item in architecture.base.LABELS]
            + [item.label for item in architecture.base.BOXES]
        )
        for required in (
            "40 cycles × 16 features",
            "k = 3",
            "k = 7",
            "k = 15",
            "2 layers · 64 channels",
            "128 channels",
            "2 layers · hidden size 64",
            "Dropout 0.4",
            "Masked MSE",
            "Soft monotonicity",
            "eligible predicted pairs",
            "L = Ldata + 0.3 Lmono",
            "ε = .005 · cmin = 300 · K = 40 · α = .2",
        ):
            self.assertIn(required, text)

    def test_drawio_is_editable_and_structurally_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "architecture.drawio"
            architecture.base.write_drawio(output)
            architecture.use_sans_serif_in_drawio(output)

            tree = ET.parse(output)
            root = tree.getroot()
            cells = root.findall(".//mxCell")
            ids = {cell.attrib.get("id") for cell in cells}
            self.assertIn("0", ids)
            self.assertIn("1", ids)
            self.assertIn("panel_a", ids)
            self.assertIn("panel_b", ids)

            edges = [cell for cell in cells if cell.attrib.get("edge") == "1"]
            self.assertGreater(len(edges), 20)
            for edge in edges:
                geometry = edge.find("mxGeometry")
                self.assertIsNotNone(geometry)
                self.assertEqual(geometry.attrib.get("relative"), "1")

            styled_text = " ".join(cell.attrib.get("style", "") for cell in cells)
            self.assertIn("fontFamily=Arial", styled_text)
            self.assertNotIn("fontFamily=Times New Roman", styled_text)


if __name__ == "__main__":
    unittest.main()
