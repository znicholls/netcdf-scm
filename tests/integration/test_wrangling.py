def test_wrangling_defaults(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
        ],
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory:" in result.output

    assert "rsut" in result.output
    assert "rlut" in result.output

    assert isdir(join(OUTPUT_DIR, "CMIP6", "one model"))
    assert isdir(join(OUTPUT_DIR, "CMIP6", "other model"))
    assert isdir(join(OUTPUT_DIR, "CMIP6", "third dir"))
    assert isdir(join(OUTPUT_DIR, "CMIP6", "fourth dir"))


def test_wrangling_var(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)
    VAR_TO_WRANGLE = ".*rsut.*"

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--variable",
            VAR_TO_WRANGLE,
            "--nested",
        ],
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory:" in result.output

    assert "rsut" in result.output
    assert "rlut" not in result.output


    assert isdir(join(OUTPUT_DIR, "CMIP6", "one model"))
    assert isdir(join(OUTPUT_DIR, "CMIP6", "other model"))


def test_wrangling_flat(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--flat",
        ],
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory:" in result.output

    assert "rsut" in result.output
    assert "rlut" in result.output

    assert len(listdir(OUTPUT_DIR)) == 3
