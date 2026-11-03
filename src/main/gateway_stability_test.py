from py4j.java_gateway import JavaGateway
import time

gateway = JavaGateway()

print("Connected to Java:", gateway.jvm.java.lang.System.getProperty("java.version"))

for i in range(100):
    assertion = (
        'org.junit.Assert.assertTrue("error message", '
        'conn.getTokenBased());'
    )

    result = gateway.entry_point.removeAssertionMessage(assertion)

    if i % 10 == 0:
        print(f"Completed: {i + 1}/100 | Result: {result}")

print("REPEATED GATEWAY TEST: PASSED")
gateway.close()
