package com.example.configs;

import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class LoggingConfig {

    public static void configureLogging() {
        Logger rootLogger = Logger.getLogger("");

        if (false) {
            rootLogger.setLevel(Level.FINEST);
        } else {
            rootLogger.setLevel(Level.OFF);
        }

        for (var handler : rootLogger.getHandlers()) {
            rootLogger.removeHandler(handler);
        }

        ConsoleHandler consoleHandler = new ConsoleHandler();
        consoleHandler.setLevel(Level.FINEST);
        consoleHandler.setFormatter(new CustomFormatter());
        rootLogger.addHandler(consoleHandler);
    }
}
