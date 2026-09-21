#include <mach-o/dyld.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef SCRIPT_NAME
#define SCRIPT_NAME "launcher.sh"
#endif

int main(void) {
    char executable[PATH_MAX];
    uint32_t size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) != 0) {
        return 2;
    }

    char executable_copy[PATH_MAX];
    if (snprintf(executable_copy, sizeof(executable_copy), "%s", executable)
        >= (int)sizeof(executable_copy)) {
        return 3;
    }
    const char *macos_dir = dirname(executable_copy);

    char candidate[PATH_MAX];
    if (snprintf(candidate, sizeof(candidate), "%s/../../../%s", macos_dir, SCRIPT_NAME)
        >= (int)sizeof(candidate)) {
        return 3;
    }

    char script[PATH_MAX];
    if (realpath(candidate, script) == NULL) {
        return 4;
    }

    execl("/bin/bash", "bash", script, (char *)NULL);
    return 5;
}
