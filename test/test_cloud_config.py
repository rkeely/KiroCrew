"""Unit tests for the cloud config store (cloud/config.py) — no secrets stored."""

from __future__ import annotations

import json

import pytest

from kiro_crew.cloud import config as cloud_config
from kiro_crew.cloud.config import (
    _MAX_FILE_BYTES,
    DEFAULT_REGION,
    CloudConfig,
    FargateConfig,
)

#: The model-credential secret as a conforming ``(name, ARN)`` pair: the name is
#: ``kirocrew/crew/<crew>/<ENV>`` and the ARN is that name plus one six-character
#: service suffix. The account id is fictional.
CREDENTIAL_SECRET = [
    "kirocrew/crew/demo/KIRO_API_KEY",
    "arn:aws:secretsmanager:us-east-1:123456789012:secret:kirocrew/crew/demo/KIRO_API_KEY-abcdef",
]

#: A complete Fargate block. Every case below is this, minus or plus one thing, so
#: a case cannot pass by being malformed in a second way the assertion never named.
COMPLETE_FARGATE = {
    "cluster": "kirocrew-crew-prod",
    "subnets": ["subnet-a", "subnet-b"],
    "security_groups": ["sg-1"],
    "image": "public.ecr.aws/example/kirocrew-crew-base@sha256:" + "a" * 64,
    "secrets": [CREDENTIAL_SECRET],
    "cpu_architecture": "X86_64",
}


class TestRespelledConstants:
    """``config.py`` respells three values from ``cloud/fargate/taskdef.py``.

    It cannot import them: the module is loaded to read a config file and must not
    pull in the AWS surface to do it. A test may, so this is the pin that keeps the
    copies equal to the originals. A drift here is a block judged complete by one
    rule and refused by another.
    """

    def test_the_architectures_match_the_engine(self):
        from kiro_crew.cloud.fargate import taskdef

        assert cloud_config._CPU_ARCHITECTURES == taskdef.CPU_ARCHITECTURES

    def test_the_digest_image_pattern_matches_the_engine(self):
        from kiro_crew.cloud.fargate import taskdef

        assert cloud_config._DIGEST_IMAGE_RE.pattern == taskdef._DIGEST_REF_RE.pattern

    def test_the_model_credential_variable_matches_the_engine(self):
        from kiro_crew.cloud.fargate import taskdef

        assert cloud_config._MODEL_CREDENTIAL_ENV == taskdef.MODEL_CREDENTIAL_ENV

    def test_config_does_not_import_the_engine(self):
        """The respelling exists because this import is forbidden."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(cloud_config))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(name.startswith("kiro_crew.cloud.fargate") for name in imported), imported


class TestFargateConfig:
    """A block is COMPLETE or ABSENT. There is no third state, by design.

    A half-written block that produced a usable object would leave the lane
    registered and refusing every launch made through it -- spending an operator's
    attention at launch time on a mistake that was visible when they saved the
    file.
    """

    def test_a_complete_block_is_read(self):
        config = FargateConfig.from_mapping(COMPLETE_FARGATE)
        assert config is not None
        assert config.cluster == "kirocrew-crew-prod"
        assert config.subnets == ("subnet-a", "subnet-b")
        assert config.security_groups == ("sg-1",)
        assert config.secrets == (
            (COMPLETE_FARGATE["secrets"][0][0], COMPLETE_FARGATE["secrets"][0][1]),
        )
        assert config.is_complete()

    @pytest.mark.parametrize(
        ("label", "block"),
        [
            ("movable tag image", {**COMPLETE_FARGATE, "image": "public.ecr.aws/x/base:latest"}),
            ("no image", {**COMPLETE_FARGATE, "image": ""}),
            ("no cluster", {**COMPLETE_FARGATE, "cluster": ""}),
            ("no subnet", {**COMPLETE_FARGATE, "subnets": []}),
            ("no security group", {**COMPLETE_FARGATE, "security_groups": []}),
            ("unknown architecture", {**COMPLETE_FARGATE, "cpu_architecture": "RISCV"}),
            ("subnets not a list", {**COMPLETE_FARGATE, "subnets": "subnet-a"}),
            ("secrets not a list", {**COMPLETE_FARGATE, "secrets": "nope"}),
            ("secret entry is not a pair", {**COMPLETE_FARGATE, "secrets": [["only-one"]]}),
            ("secret arn is empty", {**COMPLETE_FARGATE, "secrets": [["KIRO_API_KEY", ""]]}),
            ("no secrets at all", {**COMPLETE_FARGATE, "secrets": []}),
            (
                "secrets but none named for the model credential",
                {
                    **COMPLETE_FARGATE,
                    "secrets": [["kirocrew/crew/demo/OTHER_KEY", CREDENTIAL_SECRET[1]]],
                },
            ),
            ("public ip is the string false", {**COMPLETE_FARGATE, "assign_public_ip": "false"}),
            ("public ip is the string zero", {**COMPLETE_FARGATE, "assign_public_ip": "0"}),
            ("public ip is the string true", {**COMPLETE_FARGATE, "assign_public_ip": "true"}),
            ("public ip is a number", {**COMPLETE_FARGATE, "assign_public_ip": 1}),
            ("public ip is null", {**COMPLETE_FARGATE, "assign_public_ip": None}),
            (
                "secrets list past the item bound",
                {
                    **COMPLETE_FARGATE,
                    "secrets": [CREDENTIAL_SECRET] * (cloud_config._MAX_LIST_ITEMS + 1),
                },
            ),
            (
                "subnets list past the item bound",
                {**COMPLETE_FARGATE, "subnets": ["subnet-a"] * (cloud_config._MAX_LIST_ITEMS + 1)},
            ),
            (
                "image string past the size bound",
                {**COMPLETE_FARGATE, "image": "x" * (cloud_config._MAX_STRING_LEN + 1)},
            ),
            (
                "a subnet string past the size bound",
                {**COMPLETE_FARGATE, "subnets": ["s" * (cloud_config._MAX_STRING_LEN + 1)]},
            ),
            (
                "a secret name past the size bound",
                {
                    **COMPLETE_FARGATE,
                    "secrets": [
                        [
                            "x" * (cloud_config._MAX_STRING_LEN + 1) + "/KIRO_API_KEY",
                            CREDENTIAL_SECRET[1],
                        ]
                    ],
                },
            ),
            ("not an object", "nope"),
            ("absent", None),
        ],
    )
    def test_an_unusable_block_reads_as_absent(self, label: str, block: object):
        assert FargateConfig.from_mapping(block) is None, label

    def test_a_secretless_block_is_incomplete_because_the_engine_would_refuse_it(self):
        """The engine refuses a task definition delivering no model credential.

        So a block with every placement field and an empty ``secrets`` list is the
        offered-and-refusing state exactly: the lane would register and reject every
        launch. Judged here, it is absent instead.
        """
        assert FargateConfig.from_mapping({**COMPLETE_FARGATE, "secrets": []}) is None

    def test_the_credential_gate_reads_only_the_name(self):
        """The ARN-versus-name pairing is the engine's check, not this module's.

        A name whose final segment is the credential variable passes here whatever
        its ARN says; duplicating the engine's verification would be a second copy
        of a rule that must not drift.
        """
        block = {**COMPLETE_FARGATE, "secrets": [[CREDENTIAL_SECRET[0], "arn:not-checked-here"]]}
        assert FargateConfig.from_mapping(block) is not None

    @pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False)])
    def test_a_boolean_public_ip_flag_is_read_as_written(self, value: bool, expected: bool):
        config = FargateConfig.from_mapping({**COMPLETE_FARGATE, "assign_public_ip": value})
        assert config is not None
        assert config.assign_public_ip is expected

    def test_an_absent_public_ip_flag_defaults_to_false(self):
        assert "assign_public_ip" not in COMPLETE_FARGATE
        config = FargateConfig.from_mapping(COMPLETE_FARGATE)
        assert config is not None
        assert config.assign_public_ip is False

    def test_a_movable_tag_is_refused_here_rather_than_at_launch(self):
        """``taskdef.py`` requires ``<repo>@sha256:<64 hex>``.

        Caught at the file boundary, an operator learns it where they typed it. Left
        to the launch, the same mistake surfaces as a refusal from a lane they were
        offered, with nothing pointing back at ``cloud.json``.
        """
        assert FargateConfig.from_mapping({**COMPLETE_FARGATE, "image": "repo:v1"}) is None

    def test_one_bad_secret_entry_drops_the_whole_block(self):
        """Not just that entry.

        Launching with one fewer secret than the operator wrote starts a task that
        then fails on a missing variable -- which is harder to trace than a lane
        that was never offered.
        """
        two = {**COMPLETE_FARGATE, "secrets": [COMPLETE_FARGATE["secrets"][0], ["broken"]]}
        assert FargateConfig.from_mapping(two) is None

    def test_the_block_survives_a_save_and_load(self, tmp_path):
        p = tmp_path / "cloud.json"
        CloudConfig(profile="dev", fargate=COMPLETE_FARGATE).save(p)
        loaded = CloudConfig.load(p)
        assert loaded.fargate == COMPLETE_FARGATE
        assert loaded.fargate_config() is not None

    def test_an_incomplete_block_survives_an_unrelated_save(self, tmp_path):
        """Load, record a launch, save: the operator's half-written block is intact.

        This object is loaded and re-saved to record ``last_tag`` after an ordinary
        EC2 launch. Were the block judged at load and the judgement written back,
        that save would replace a block the operator is mid-way through editing
        with ``null`` -- an incomplete block must read as ABSENT to the lane and
        still round-trip through the file unchanged.
        """
        p = tmp_path / "cloud.json"
        half_written = {"cluster": "kirocrew-crew-prod", "subnets": ["subnet-a"]}
        p.write_text(
            json.dumps({"profile": "dev", "region": "us-west-2", "fargate": half_written}),
            encoding="utf-8",
        )
        cfg = CloudConfig.load(p)
        assert cfg.fargate_config() is None
        cfg.last_tag = "kc-after-ec2"
        cfg.save(p)
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        assert on_disk["fargate"] == half_written
        assert on_disk["last_tag"] == "kc-after-ec2"
        assert CloudConfig.load(p).fargate_config() is None

    @pytest.mark.parametrize("block", ["a string", 42, ["a", "list"], {"secrets": "nope"}])
    def test_a_malformed_block_also_survives_a_save(self, tmp_path, block: object):
        """Not only an incomplete object: any shape the file held is written back."""
        p = tmp_path / "cloud.json"
        p.write_text(json.dumps({"profile": "dev", "fargate": block}), encoding="utf-8")
        cfg = CloudConfig.load(p)
        assert cfg.fargate_config() is None
        cfg.save(p)
        assert json.loads(p.read_text(encoding="utf-8"))["fargate"] == block

    def test_a_corrupt_block_leaves_the_rest_of_the_config_readable(self, tmp_path):
        """One bad block must not cost the profile and region too."""
        p = tmp_path / "cloud.json"
        p.write_text(
            json.dumps({"profile": "dev", "region": "us-west-2", "fargate": {"cluster": "c"}}),
            encoding="utf-8",
        )
        loaded = CloudConfig.load(p)
        assert loaded.fargate_config() is None
        assert loaded.profile == "dev"
        assert loaded.region == "us-west-2"

    def test_no_secret_value_field_exists_to_write_one_into(self):
        """The file's contract is identifiers only, and the shape enforces it.

        A secret is named by its canonical name and its ARN; the value is fetched by
        the task's execution role before the container starts. A field for a value
        would be the first place someone put one.
        """
        fields = {f.name for f in FargateConfig.__dataclass_fields__.values()}
        for forbidden in ("secret_values", "value", "values", "password", "token"):
            assert forbidden not in fields


class TestCloudConfig:
    def test_defaults(self, tmp_path):
        cfg = CloudConfig.load(tmp_path / "cloud.json")
        assert cfg.profile == ""
        assert cfg.region == DEFAULT_REGION
        assert cfg.last_tag == ""

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "cloud.json"
        cfg = CloudConfig(profile="dev", region="us-west-2", last_tag="kc-abc")
        cfg.save(p)
        loaded = CloudConfig.load(p)
        assert loaded.profile == "dev"
        assert loaded.region == "us-west-2"
        assert loaded.last_tag == "kc-abc"

    def test_over_long_last_tag_sanitized_to_empty(self, tmp_path):
        # A 52-63 char last_tag must be sanitized to "" on load — NOT carried
        # into the resume path where validate_tag (cap 51) would raise. The
        # sanitizer's job is "no last launch", not a crash. Keep _TAG_RE in
        # lockstep with ec2._TAG_RE.
        from kiro_crew.cloud import ec2

        assert ec2._TAG_RE.pattern == r"^[a-zA-Z0-9-]{1,51}$"  # the cap we mirror
        p = tmp_path / "cloud.json"
        p.write_text('{"profile": "dev", "region": "us-east-1", "last_tag": "%s"}' % ("a" * 60))
        cfg = CloudConfig.load(p)
        assert cfg.last_tag == ""  # too long -> dropped, no ValidationError later
        # A malformed-charset tag is likewise dropped.
        p.write_text('{"last_tag": "bad tag!"}')
        assert CloudConfig.load(p).last_tag == ""
        # A valid 51-char tag is kept.
        p.write_text('{"last_tag": "%s"}' % ("k" * 51))
        assert CloudConfig.load(p).last_tag == "k" * 51

    def test_never_stores_credentials(self, tmp_path):
        p = tmp_path / "cloud.json"
        CloudConfig(profile="dev", region="us-east-1", last_tag="t").save(p)
        text = p.read_text(encoding="utf-8")
        # Only profile/region/last_tag — no secret-shaped keys.
        for forbidden in ("secret", "access_key", "aws_access", "token", "password"):
            assert forbidden not in text.lower()

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        p = tmp_path / "cloud.json"
        p.write_text("not json{{{")
        cfg = CloudConfig.load(p)
        assert cfg.region == DEFAULT_REGION

    def test_non_object_json_falls_back_to_defaults(self, tmp_path):
        # Valid JSON that isn't an object ("hello", [1,2], 42, null) must not
        # raise AttributeError out of load() — that would give a raw traceback on
        # every `kirocrew cloud` command (handle_cloud only catches AWS/validation
        # errors). Honor the tolerate-a-corrupt-file promise.
        for body in ('"hello"', "[1, 2, 3]", "42", "null", "true"):
            p = tmp_path / "cloud.json"
            p.write_text(body)
            cfg = CloudConfig.load(p)
            assert cfg.region == DEFAULT_REGION
            assert cfg.profile == ""
            assert cfg.last_tag == ""

    def test_missing_region_coerced_to_default(self, tmp_path):
        p = tmp_path / "cloud.json"
        p.write_text('{"profile": "dev", "region": ""}')
        cfg = CloudConfig.load(p)
        assert cfg.region == DEFAULT_REGION


class TestCloudConfigIsSealedAgainstAgentWrites:
    """``cloud.json`` must not be writable through an agent's file-edit tool.

    The Fargate block made this file an input to a security decision:
    ``fargate.image`` chooses the container a launch runs, and the task's execution
    role delivers the model credential into that container. An agent that could
    rewrite the field could name a digest-pinned image of its own -- the digest rule
    constrains the reference's FORM, not who owns the registry -- while leaving the
    owner's placement and secrets intact, so the owner's next launch would hand the
    credential to an image they never chose.
    """

    #: Spelled out rather than looped from the production tuple: a test derived from
    #: the same list the code reads would keep passing after someone emptied it.
    PATHS = ("~/.kiro/crew/cloud.json", "~/.kirocrew/cloud.json")

    @pytest.mark.parametrize("path", PATHS)
    def test_an_agent_may_not_write_it(self, path: str):
        from kiro_crew.security.paths import is_sensitive_write_path

        assert is_sensitive_write_path(path) is True

    @pytest.mark.parametrize("path", PATHS)
    def test_it_stays_readable(self, path: str):
        """Write-protected, not sensitive: the selector reads it on every request.

        Blocking reads would take the Set-up tab down instead of protecting it.
        """
        from kiro_crew.security.paths import is_sensitive_path

        assert is_sensitive_path(path) is False

    def test_the_predicate_can_still_say_no(self):
        """Positive control, so the two assertions above cannot pass vacuously.

        A sibling path under the same crew home that nothing protects must come back
        writable; without this, a predicate that answered True for everything would
        satisfy this class.
        """
        from kiro_crew.security.paths import is_sensitive_write_path

        assert is_sensitive_write_path("~/.kiro/crew/not-a-protected-leaf.json") is False

    def test_the_gateway_can_still_save_it(self, tmp_path):
        """Sealing is placement, not logic: Kiro Crew's own write is unaffected.

        ``save()`` is how an ordinary EC2 launch records ``last_tag``, so a seal that
        also stopped the gateway would break the lane it exists to protect.
        """
        p = tmp_path / "cloud.json"
        CloudConfig(profile="dev", last_tag="t1").save(p)
        assert CloudConfig.load(p).last_tag == "t1"


class TestSaveRefusesWhatLoadWouldReject:
    """A save that outlives its own reader is silent config loss, so it is refused.

    The raw ``fargate`` block is stored verbatim and ``save()`` re-serializes with
    ``indent=2``, so a hand-minified block that fits under the read ceiling can cross
    it once expanded. Publishing that file would make the next ``load()`` fall back to
    defaults and lose the operator's profile, region and Fargate state with nothing
    said. Refusing keeps the previous, readable file.
    """

    def test_an_ordinary_record_still_saves(self, tmp_path):
        p = tmp_path / "cloud.json"
        CloudConfig(profile="dev", region="us-east-1", last_tag="t1").save(p)
        assert CloudConfig.load(p).profile == "dev"

    def test_a_record_that_would_outgrow_the_read_ceiling_is_refused(self, tmp_path):
        p = tmp_path / "cloud.json"
        CloudConfig(profile="dev").save(p)
        before = p.read_text(encoding="utf-8")
        oversized = CloudConfig(profile="dev", fargate={"pad": "x" * (_MAX_FILE_BYTES + 10)})
        with pytest.raises(ValueError, match="unreadable"):
            oversized.save(p)
        assert p.read_text(encoding="utf-8") == before, "the readable file must survive"

    def test_the_refusal_matches_what_load_enforces(self, tmp_path):
        """The two ceilings are the same constant, not two numbers that can disagree."""
        p = tmp_path / "cloud.json"
        p.write_text("x" * (_MAX_FILE_BYTES + 1), encoding="utf-8")
        assert CloudConfig.load(p).profile == ""


class TestTheSealedCloudConfigNameCannotBeAnAlias:
    """A bind mount seals a link's REFERENT, so the sealed leaf must not be a link.

    Otherwise the lexical name stays replaceable in a writable parent: a sandboxed
    process unlinks it, drops its own file there, and the seal is intact around a name
    that now chooses which image a Fargate launch runs.
    """

    def test_cloud_json_is_on_the_nofollow_file_list(self):
        from kiro_crew import sandbox

        assert "cloud.json" in sandbox._CREW_NOFOLLOW_READONLY_FILE_LEAVES

    def test_every_nofollow_file_leaf_is_also_precreated(self):
        """Mirrors the assert the directory list carries.

        A nofollow leaf that is not materialised has no seal to protect on a default
        install, so the strict check would guard a name nothing binds.
        """
        from kiro_crew import sandbox

        assert set(sandbox._CREW_NOFOLLOW_READONLY_FILE_LEAVES) <= set(
            sandbox._CREW_PRECREATE_READONLY_FILE_LEAVES
        )

    def test_the_list_is_not_everything(self):
        """Positive control: the strict path is opt-in, not applied to every leaf."""
        from kiro_crew import sandbox

        assert set(sandbox._CREW_PRECREATE_READONLY_FILE_LEAVES) - set(
            sandbox._CREW_NOFOLLOW_READONLY_FILE_LEAVES
        )


class TestTheStrictCloudConfigSealRefusesEveryUncoveredName:
    """A read-only bind seals a MOUNT, not an inode, so two names escape it.

    A symlink leaves the lexical name replaceable; a second hardlink puts an alias
    outside the mount that reaches the same inode. For an ordinary ceiling the codebase
    only WARNS about both, because refusing would break a dotfile manager or a snapshot
    tool. For ``cloud.json`` a write through either name picks the container image a
    Fargate launch runs, and the execution role delivers the model credential into it,
    so the strict leaf refuses where the rest warn.
    """

    @staticmethod
    def _strict_target(tmp_path):
        from kiro_crew import sandbox

        return tmp_path / sandbox._CREW_NOFOLLOW_READONLY_FILE_LEAVES[0]

    def test_a_lone_regular_file_is_accepted(self, tmp_path):
        from kiro_crew import sandbox

        p = self._strict_target(tmp_path)
        p.write_text("{}", encoding="utf-8")
        sandbox._require_real_file_nofollow(str(p))

    def test_an_absent_file_is_accepted(self, tmp_path):
        """Absent is the publish path's job, not this check's."""
        from kiro_crew import sandbox

        sandbox._require_real_file_nofollow(str(self._strict_target(tmp_path)))

    def test_a_symlink_is_refused(self, tmp_path):
        from kiro_crew import sandbox

        real = tmp_path / "elsewhere.json"
        real.write_text("{}", encoding="utf-8")
        p = self._strict_target(tmp_path)
        p.symlink_to(real)
        with pytest.raises(sandbox.SandboxCeilingUnsealable, match="SYMLINK"):
            sandbox._require_real_file_nofollow(str(p))

    def test_a_hardlinked_file_is_refused(self, tmp_path):
        """The shape the earlier symlink-only check missed entirely."""
        from kiro_crew import sandbox

        p = self._strict_target(tmp_path)
        p.write_text("{}", encoding="utf-8")
        (tmp_path / "alias.json").hardlink_to(p)
        with pytest.raises(sandbox.SandboxCeilingUnsealable, match="hardlink"):
            sandbox._require_real_file_nofollow(str(p))

    def test_a_directory_at_the_name_is_refused(self, tmp_path):
        from kiro_crew import sandbox

        p = self._strict_target(tmp_path)
        p.mkdir()
        with pytest.raises(sandbox.SandboxCeilingUnsealable, match="not a regular file"):
            sandbox._require_real_file_nofollow(str(p))


class TestTheCredentialGateAgreesWithTheEngine:
    """No spelling may pass this gate and then be refused at launch.

    The gate exists so an incomplete block leaves the lane UNREGISTERED instead of
    registered-and-refusing. A gate that accepts a name the engine rejects recreates
    that exact state from inside the gate, which is what a tail-only check did: both a
    bare ``KIRO_API_KEY`` and a wrong-prefix ``junk/KIRO_API_KEY`` passed here and were
    refused by ``identity.secret_env_name``.
    """

    @staticmethod
    def _with_credential_named(name: str) -> dict:
        return {
            **COMPLETE_FARGATE,
            "secrets": [
                [name, f"arn:aws:secretsmanager:us-east-1:123456789012:secret:{name}-AbCdEf"]
            ],
        }

    @pytest.mark.parametrize(
        "name",
        [
            "KIRO_API_KEY",
            "junk/KIRO_API_KEY",
            "kirocrew/crew/KIRO_API_KEY",
            "kirocrew/crew//KIRO_API_KEY",
            "kirocrew/crew/a/b/KIRO_API_KEY",
            "kirocrew/crew/demo/OTHER_KEY",
        ],
    )
    def test_a_name_the_engine_would_refuse_does_not_register_the_lane(self, name: str):
        assert FargateConfig.from_mapping(self._with_credential_named(name)) is None, name

    def test_a_conforming_name_is_accepted(self):
        block = self._with_credential_named("kirocrew/crew/demo/KIRO_API_KEY")
        assert FargateConfig.from_mapping(block) is not None

    def test_every_name_this_gate_accepts_the_engine_also_accepts(self):
        """The property itself, checked against the engine rather than restated.

        Any name this module lets through must survive ``secret_env_name``, which is
        what the launch path calls. Run over the same spellings as above so the two
        halves cannot drift apart silently.
        """
        from kiro_crew.cloud.fargate.identity import SecretRef, secret_env_name

        names = [
            "kirocrew/crew/demo/KIRO_API_KEY",
            "KIRO_API_KEY",
            "junk/KIRO_API_KEY",
            "kirocrew/crew//KIRO_API_KEY",
            "kirocrew/crew/a/b/KIRO_API_KEY",
        ]
        for name in names:
            block = self._with_credential_named(name)
            if FargateConfig.from_mapping(block) is None:
                continue
            arn = f"arn:aws:secretsmanager:us-east-1:123456789012:secret:{name}-AbCdEf"
            secret_env_name(SecretRef(name=name, arn=arn))

    def test_the_prefix_matches_the_engines(self):
        """The third respelled constant, pinned like the other two."""
        from kiro_crew.cloud import config as cloud_config
        from kiro_crew.cloud.fargate import identity

        assert cloud_config._SECRET_NAME_PREFIX == identity.SECRET_NAME_PREFIX
