from .runner import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Derby one_camp_n_days experiment from YAML config.")
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default=None,
                        help='Optional directory to write epoch-level parquet logs')
    parser.add_argument('--log-level', dest='log_level', default='INFO', type=str,
                        help='Logging verbosity for this run (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE). Not propagated to inner APIs.')
    parser.add_argument('--flush-every', dest='flush_every', default=1, type=int,
                        help='Number of epochs between parquet flushes (default 1).')
    args = parser.parse_args()
    main(
        args.config,
        output_dir=args.output_dir,
        log_level=args.log_level,
        flush_every=args.flush_every,
    )
