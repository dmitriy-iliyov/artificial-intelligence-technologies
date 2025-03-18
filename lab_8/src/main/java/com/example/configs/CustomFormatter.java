package com.example.configs;


import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;

public class CustomFormatter extends Formatter {

    @Override
    public String format(LogRecord record) {
        if (record.getLevel() == Level.INFO)
            return String.format("\u001B[36m%s: ", record.getLevel().getName()) + String.format("\u001B[37m%s\n", record.getMessage());
        if (record.getLevel() == Level.CONFIG)
            return String.format("\u001B[33m%s: ", record.getLevel().getName()) + String.format("\u001B[37m%s\n", record.getMessage());
        if (record.getLevel() == Level.FINEST) {
            return String.format("\u001B[32m%s: ", "DEBUG") + String.format("\u001B[37m%s\n", record.getMessage());
        }
        return "";
    }
}
