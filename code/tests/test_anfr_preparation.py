import tempfile
import unittest
from pathlib import Path

from pyl_poil.anfr import prepare_department_reference


class AnfrPreparationTests(unittest.TestCase):
    def test_prepare_department_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "SUP_SUPPORT.txt"
            nature = root / "SUP_NATURE.txt"
            support.write_text(
                "SUP_ID;STA_NM_ANFR;NAT_ID;COR_NB_DG_LAT;COR_NB_MN_LAT;COR_NB_SC_LAT;COR_CD_NS_LAT;"
                "COR_NB_DG_LON;COR_NB_MN_LON;COR_NB_SC_LON;COR_CD_EW_LON;SUP_NM_HAUT;ADR_LB_LIEU;ADR_LB_ADD1;ADR_NM_CP;COM_CD_INSEE\n"
                "1;780010001;23;48;48;0;N;2;8;0;E;30;;;78000;78000\n"
                "1;780010002;23;48;48;0;N;2;8;0;E;30;;;78000;78000\n"
                "2;920010001;23;48;50;0;N;2;10;0;E;20;;;92000;92000\n",
                encoding="utf-8",
            )
            nature.write_text("NAT_ID;NAT_LB_NOM\n23;Pylône autostable\n", encoding="utf-8")
            df = prepare_department_reference(support, nature, "78")
            self.assertEqual(len(df), 1)
            self.assertEqual(str(df.iloc[0]["SUP_ID"]), "1")
            self.assertEqual(int(df.iloc[0]["NB_STATIONS_ANFR"]), 2)
            self.assertAlmostEqual(float(df.iloc[0]["latitude"]), 48.8, places=6)
            self.assertAlmostEqual(float(df.iloc[0]["longitude"]), 2.1333333333, places=6)


if __name__ == "__main__":
    unittest.main()
