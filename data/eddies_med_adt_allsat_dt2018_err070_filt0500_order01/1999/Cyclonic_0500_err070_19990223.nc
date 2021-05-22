CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�� ě��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N+�   max       P��2       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��{   max       <e`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F�=p��
       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vhQ��       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�l`           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       <D��       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��i   max       B1��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��    max       B0�`       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >5��   max       C�&�       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =���   max       C�(       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          I       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N+�   max       PwU�       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?�3��ߤ       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <e`B       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F�=p��
       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vg�
=p�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�`           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F   max         F       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}}�H˓   max       ?�3��ߤ     0  ^            	      I   6   &   0         
         	   @         
      +   
               
   A   	                                     $   3                           
   	   @               	   *               
                      /O#q�N+�O<��Nu2xN��PK6�P��O�j�P =3OR��OϘ	N)��O	��N�~RN&��P��2Oe*�N��Of��N#�O�8�Nqg�O]�eN7��NGI�O�y�N��RO�M2N,�{O�kOy��OecN�J�N��O���No��Oǹ�O�cN�m�N��O�vZP:�P��N���P
�N�O�OI[uN�Y�N.@NǤ8O5,�Pc��NvAO2�N�-�N��|N��P/��O�.�N/0�O���NU"�NCAnO�O��eN��(N���O]�NA0�O��<e`B<e`B<T��<#�
<t�;��
;�o:�o%   ��`B�o�t��#�
�#�
�D���D���u��o��o��C���C����㼴9X��9X��j��j��j�ě��ě����ͼ��ͼ��������������������C��C��\)�\)�\)�\)��w��w��w�#�
�''',1�0 Ž0 Ž8Q�D���P�`�T���T���]/�aG��ixսixսu��7L��\)��\)��hs�������{��$!&��������������������������#%/<KRPOSHB</'#����������������������
#/03/,#
���5B[t�������t[5)joz�����������zlggj��#/<IHBD7/#
�����'/HU`de^SH<#
����������������������������������������46BOSZOB964444444444��������������������rtz�������������ttrr)6?:6)
#<��������{U<0
���������������|||��y����������ywvyyyyyy��������������������mnoz����znmmmmmmmmmm�����������������~�����������������������	"))*("	����������


���������� "".//442/)""      ��������������������������������������������
#,'%'(#
����� #/242/#"           ����������������������������������������������������������������������������))*.240)$#$&))))))))��������������������)/0<HPTRH=</))))))))&*4C\h������u\C60(&������!�����������������������������������������������������������^cnz���������zlha][^����������������������������������������������������������������������������������������������������=IUbnptxxzxncaUNI?<=���������������������������������������-005<AISU_UMI<0-'&)-����������������������9CLN[ee`P5�����ltx�������utllllllllMNP[gt~�����ytgb[UNMABLO[hhh_][XVOIB?<AA"#$0<=IKNOLIB<60)$#"259<@BINZTPTVWNB:542DM[g����������tgN><D������znaUOKKUaz������������������������eiq��������������yne#.0010(#����������������������������������� (*)���������������������������	

����������������������������������������������)>A8,����������������������������������������������	� ��	����"�#�"���	�	�	�	�	�	�	�	�[�O�B�8�6�2�2�6�B�O�[�h�n�t�y�t�h�`�a�[�����������������������������t�p�g�e�\�\�g�t�z�ѿ������������������ݿ��� �����������������������/�B�_�wā�t�[�O�B�)��à×ÖÓÙ×ìóù����������������ùìà�A�4�#�����(�4�A�Z�s�~�z�����|�s�Z�A�U�M�H�<�3�1�4�<�H�U�h�zÅ�u�w�n�k�o�a�U�y�`�I�B�A�G�T�[�W�^�l�������Ŀۿݿ̿��y�a�`�X�]�a�n�y�s�n�e�a�a�a�a�a�a�a�a�a�a�g�c�Z�Q�N�F�I�N�Y�Z�g�i�s��������s�i�g���������������������
������
������ܹڹֹعܹ������ܹܹܹܹܹܹܹܹܹ��������a�J�(�!�%�N���������������������������������	��$�/�;�H�R�Q�H�;�/�"��	������%�(�4�A�M�W�M�L�A�4�(�������h�k�s����������������Ǿƾľ���������h�U�N�H�B�@�H�U�U�W�U�U�U�U�U�U�U�U�U�U�U�'������6�M�Y�f�����������f�Y�M�'�����������&������������	�������������������	��#�-�/�3�/�"��	����پ׾ξ׾����������������������������������������������������������������лܻ������������ܻû�àÚÓÓÇÉÓØàæìùû������ùìààECE*EEEEE*EPE\EiEuE�E�E�E~E�E�E�EuECE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���
���� ����
��#�5�H�U�^�X�M�H�/�#��_�F�B�:�2�/�:�F�P�_�l�x�����������x�l�_�b�_�U�Q�Q�U�]�b�n�x�{ŇŔŞŔŎŇ�z�n�b��
�������"�*�/�0�*��������s�k�s�����������������s�s�s�s�s�s�s�s�	���������	��"�.�J�T�]�T�G�;�.���	����������������������������������������ʾ����������������ʾ׾����������ʺ�����ۺպӺӺպֺ������������������������{���������������������������������}�������������������������������������ѿĿ������������ĿѿԿݿ����������ѹ����~�z��������ùܺ�������۹Ϲù��3�(�3�C�L�Y�~���ɺֺ����ֺɺ��~�L�3��
�������'�4�>�9�4�'��������O�C�'����6�A�O�\�hƎƲ����ƸƳƧƎ�O���������������������������������������һ��������������ûлܻ����ܻлû»����û������������ûлܻ���������ܻл��������������������������������������������������������������������������������׺���������!�$�-�8�:�:�8�1�-�!�����������������������������ʼּؼмʼ��������f�^�g�����������������	�
���������������������������������������������������������������������������� �������������������������������ûɻû������������������������(�/�4�7�4�/�(���ŔōŇ�{�n�f�b�^�b�n�{ŇŔŠŦŪŭŮŭŔ¿¦¦¿������������������¿�������Ŀ��������ݿѿĿ�����������������������������������������������������čā�t�g�^�]�h�tāĦĳ��������ĿĳĦĚč���������������������ĽɽĽ����������������
�����������
������������������������������ʼּ��ۼּҼʼƼ����X�B�*�6�B�C�O�[�h�t�wāĐĚĤęčā�h�X������$�0�=�>�=�:�0�$���������ܻлû��������������ûл߻����������$����$�0�=�I�V�Z�\�X�V�I�=�0�$�$�$�$��߹ܹ׹ܹ����� ���������������������ÿöóòòù������������������ + N ( S a > @ G ' s � [  P I > P _ Z f < 1 ] a l g y H ^ 8 B ? 1 � @ Z X k ~ e @ G s T ` k @ % q W h L A ? ) X  h ) ` m g ; l G F ` � < 7 T    `  2  �  �    �  �    9    �  j  ,  �  Z  �  �  �    b     z     o  �  �  �  �  C  �  �  6  v  �  P  �    �  �  �  �  �  }  �  $  J  $  �  �  [     �  �  �  z  �  	  P  �  �  �  �  l  �  S  M  �  W  $  V  }�D��<D���t�:�o�o��7L�L�ͽt��@��+���
����ě��T�����
�����P�ě������ě��u��h�C���/���ͽ,1�o��9X�+�@��ixս0 Žo��h�D���t��u�49X���#�
��\)�� Ž���#�
��C��'�7L�T���<j�8Q�P�`�L�ͽ���D���}�u��O߽y�#��vɽ��T�}󶽡����C���O߽\������w���w�\�� ž+Bx`B�OB��B�VB]�B��B ��B��B�LB ߤB+@�BU�BS1B?.BP�B'�vB ��B�B ��B[�B AB��A��iBM�A�v�B0�B\�B��B�B��B��B�B35B0�B��B�@B1��B��B��B;YB��B�LB�B)�5B��B;�B�B'�8B6kB�yB&g@B+�CB{cB
;kB	d`B��B&!�B�B	�B�*BbgB
�B%&�BxzB/jB1nB%$B$%B@B!9 B�*BArB�B�B�wB@�B	3�B �sB�bB@rB �3B*��BA?B�BPB@�B(D�B ��B�JB!GGB��B FnB��A�� B��A��qB�pB�8B"�B9]BvB� B �"B?$B"�BDB��B0�`B;B�B>�B��B�B2B)�B<�BD�B��B'\�B>XB�DB&0iB,B�BAUB
z�B	B?B��B&@B��B	�B��B��B
�B%@B�_B)�B?|B�B$��B<B!1�B�mA�:A�&�A��wA�C�A�!A{o�A�B?Aͭ	A<4�A�1�Al�0A��A��A�l�>�n�A���A��A8��AI�(A�2@�NZA�TA��,AUG�A���@��A�,C���C�&�A�ƞ@���A��A�9�AG�A^�A���AR��@Go�A��A�`AA{�>5��@c�@�D�B�}A��@��K@��A�z�A��u@g{)@�3�A�E!A��kA�%@��	A3�,A�	A�� Az�bA�A�Aޓ�A$�A��@���A�cQB	�j@���B
�e?uA�g�A� �A��^A�uLA�p�A�P$Aza�A��gÀ�A<�A�YAmAƁ�A�{�A���? &�A���A��lA7:GAL��A�|�@وEA�|�A���AT�A���@��VA̩�C��C�(A�E@��5A���A�aqAH��A\�A���AR��@C�#A��mA��(Az��=���?�±@Ŗ!B�XA��8@�A�@���A�r�A�k�@k��@��CA�'�A��A�}�@��A4��A�eA���A{�A��A݂VA#�A�y@���A�J�B	�4@�
B
�?��A�             
      I   7   &   0                  	   A      	         ,                  
   A   
      !                              $   4                              	   @               
   *               
          	            0                  -   '   #   #      -               ;               '               #                                 #            !   '   +      -                        1                  )          !                                             #                              5               !               #                                 #            !      +      )                        1                  '                                    O#q�N+�N���NMT�N��\P�zO�"O�NO
XtNy�!O=k�N)��N�:�N`��N&��PwU�OSf�N[d�Of��N#�O��Nqg�O]�eN7��NGI�O�y�N��RO7�N!CO�kO0�OecN�J�N��OZoNo��Oǹ�N��,N�m�N��O���O� ?P��N���P�EN�O�N�dN+��N.@NǤ8O5,�Pc��NvAO%7N0&�N��|N��P)EMOU�N/0�O\�mNU"�NCAnO�Omt�N��(N���N�"�NA0�O�<�    �  p  �  �  �  u  �  �  X  G  �    d    �  �  �  �  p  S  (  [    �  �  �    u  5  C  �  �    �  �  �  <  <  7  $     �  (  ~  �  ^  o  �  �  �    	-  �  d  �  Z  �  z  �  #  �    �  C  �  �    v  v  <e`B<e`B<o<t�<o�u�#�
�o�������
�D���t��T���49X�D�����
��o��C���o��C��ě����㼴9X��9X��j��j��j�P�`�������ͼ��������������+�����+�C��C���P�P�`�\)�\)�#�
��w��w�49X�,1�'',1�0 Ž0 Ž<j�P�`�P�`�T���Y���%�aG��q���ixսu��7L�����\)��hs���㽧� ���$!&��������������������������!#-/<@HMKKH=<:/,&#!��������������������
#-/2/+#
�(5BN[t������t[N5)(moz������������zsnmm���
#/:66/,#
�����##)/<HRUWUSHE</###����������������������������������������46BOSZOB964444444444��������������������w����������zwwwwwwww)6?:6)#0In�������{U<0#����������������~|��z����������{xwzzzzzz��������������������mnoz����znmmmmmmmmmm������������������������������������������	"))*("	����������


���������� "".//442/)""      ������������������������������������������
 ##
������!#/131/#"!!!!!!!!!!!����������������������������������������������������������������������������))*.240)$#$&))))))))��������������������)/0<HPTRH=</))))))))&*4C\h������u\C60(&���� ������������������������������������������������������

������joz����������zxtljij����������������������������������������������������������������������������������������������������DIQU`bcnnqrrnjbUIHCD����������������������������������������-005<AISU_UMI<0-'&)-����������������������9CLN[ee`P5�����ltx�������utllllllllNNQ[gt{�����vtgd[UNN?BHOVYUSOLBB????????"#$0<=IKNOLIB<60)$#"259<@BINZTPTVWNB:542EN[g����������tgN?=ERUVanz������zneaWUR��������������������glr��������������{og#.0010(#������������������������������������%'���������������������������	

����������������������������������������������)<@7+����������������������������������������������	� ��	����"�#�"���	�	�	�	�	�	�	�	�B�@�6�5�5�6�;�B�O�[�h�t�u�t�m�h�[�W�O�B��������������� ����������������������t�q�g�g�_�g�j�t�x�t�t�t�t�Ŀ��������������ѿݿ��������������ѿ��������������)�B�J�b�b�[�O�B�6�)��àßáÞÝÜâìù����������������ùìà�A�@�4�/�.�-�4�:�A�I�M�W�Z�b�e�a�Z�M�A�A�H�H�?�?�H�U�a�b�c�a�\�U�H�H�H�H�H�H�H�H�y�m�`�\�O�G�F�G�T�`�m�y���������������y�a�`�X�]�a�n�y�s�n�e�a�a�a�a�a�a�a�a�a�a�g�\�Z�N�O�Z�g�s��������z�s�g�g�g�g�g�g���������������
����
�����������������ܹڹֹعܹ������ܹܹܹܹܹܹܹܹܹ����j�Z�I�9�5�B�Z����������������������������������� �	��"�/�;�H�P�L�H�;�/��	������(�,�4�A�M�M�M�J�A�4�(�������h�k�s����������������Ǿƾľ���������h�U�N�H�B�@�H�U�U�W�U�U�U�U�U�U�U�U�U�U�U�"��� �'�@�M�Y�f�����������f�Y�M�4�"�����������&������������	�������������������	��#�-�/�3�/�"��	����پ׾ξ׾����������������������������������������������������������������лܻ������������ܻû�àÚÓÓÇÉÓØàæìùû������ùìààE7E5E*E&E*E-E7ECEPEQE\EiEiEmEmEiE\EPECE7E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���
���� ����
��#�5�H�U�^�X�M�H�/�#��l�_�S�J�F�?�E�F�S�]�b�x�������������x�l�b�_�U�Q�Q�U�]�b�n�x�{ŇŔŞŔŎŇ�z�n�b��
�������"�*�/�0�*��������s�k�s�����������������s�s�s�s�s�s�s�s�	����������	��"�.�7�G�O�J�G�<�;�.��	����������������������������������������ʾ����������������ʾ׾����������ʺ�޺ٺֺԺպֺ׺����������������������������{���������������������������������}��������������������������������������ѿĿ������������Ŀѿݿ����� ����깝���������������ùϹ۹��������Ϲù����3�(�3�C�L�Y�~���ɺֺ����ֺɺ��~�L�3��
�������'�4�>�9�4�'��������O�C�*��!�6�C�O�\�hƎƧƱ����ƳƧƚƎ�O���������������������������������������һ��������������ûлܻ����ܻлû»����û����������������ûлܻ�����ݻܻл��������������������������������������������������������������������������������׺���������!�$�-�8�:�:�8�1�-�!�����������������������������ʼּؼмʼ��������f�^�g�����������������	�
�������������������������������������������������������������������������������������������������ûƻû����������������������������������(�/�4�7�4�/�(���ŔōŇ�{�n�f�b�^�b�n�{ŇŔŠŦŪŭŮŭŔ¿¦¦¿������������������¿�ѿǿĿ����������Ŀѿݿ�������ݿٿ�����������������������������������������čā�t�h�_�_�h�tāĚĦįĳ��ĿĳĮĦĚč���������������������ĽɽĽ����������������
�����������
������������������������������ʼּ��ۼּҼʼƼ����h�_�[�O�C�F�O�Q�[�h�tāčĚġĚĖčā�h������$�0�=�>�=�:�0�$���������ܻлû��������������ûл߻����������$�!���$�0�=�I�V�Y�[�W�V�I�=�0�$�$�$�$��߹ܹ׹ܹ����� ���������������������ÿ÷ôòòù������������
������� + N ) U \ ? ? : 8 6 t [ ! g I 3 N ] Z f 5 1 ] a l g y < R 8 0 ? 1 � 4 Z X _ ~ e : < s T [ k @ $ v W h L A ? ( <  h * : m > ; l G 7 ` � : 7 T    `  2    J  �  \  �    7  �  �  j  �  �  Z  �  �  �    b  �  z     o  �  �  �  ,  $  �  �  6  v  �  �  �      �  �  @  P  }  �  �  J  $    r  [     �  �  �  `  L  	  P  �  a  �  �  l  �  S  �  �  W    V  n  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F               �  �  �  �  �  �  j  ,  �  �  :  �  �    �  �  �  �  �  �  �  ~  {  w  p  e  Z  O  D  9  .  #      9  ?  @  ?  ]  c  E    �  �  �  �  �  ~  n  _  K  1  �  t  �  �  �  �  �  �  �  �  �  �  �  q  R  4    �  �  �  M  �  k  z  �  v  i  Z  J  9  %    �  �  �  �  �  �  �  �  L    �    L  o  �  �  �  }  S    �  �  ,  �  1  �    h  f  �  �    ;  ]  p  u  i  R  -  �  �  v     �  y  �  v  �  .  Y  2  k  �  �  �  �  �  �  �  �  h  H  	  �  T  �  a  �    v  �  N  �  �  �  0  c  �  �  �  �  �  �  �  �  �  G  g  �   �  )  P  �  �    7  E  M  I  H  Q  *  �  �  �  O    �  �  �  2  3  1  *  #  $  >  D  ?  .      �  �  �  �  `  8   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  7    �  �  M    �     	        �  �  �  �  �  �  �  h  D    �  �  e  $  G  K  P  U  Y  ^  b  S  ;  "  	  �  �  �  �  �  �  v  c  P    v  l  b  X  L  <  +    
  �  �  �  �  �  �  �  w  c  P  �  �  �  �  �  �  �  �  u  T  $  �  �  e    �  �  ?  �   �  �  �  �  �  �  �  �  �  �  �  |  V  '  �  �  x  >    �  U  �  �  �  �  �  �  �  �  �  �  �  |  d  J  /    �  �  �  |  �  �  �  �  �  w  f  \  U  Q  G  7  !    �  �  �  �  �  �  p  h  `  T  ?  *    �  �  �  �  �  x  ^  D  +      2  I  �  -  F  S  R  D  2  $    1  &    �  �  ]  �  X  �  �  #  (  !      �  �  �  �  �  d  ?    �  �  �  I    �  �  L  [  T  M  D  9  ,         �  �  �  �  s  @    �  w     w          �  �  �  �  �  �  �  �  �    ^  =    �  �  �  �  �  |  v  p  j  d  ^  X  R  J  @  6  -  #         �   �  �  �  �  �  �  x  \  <    �  �  t  6  �  �  �  �  v  x  m  �  �  p  U  :    �  �  �  �  �  u  e  O  *    �  �  s  6  	m  	�  
h  
�  
�  
�  
�  
    
�  
�  
�  
�  
>  	�  	q  �  g  �  �  t  t  s  t  t  s  q  n  k  f  `  [  V  R  O  I  ?  +  �  U  5  
  �  �  z  C  
  �  �  \  #  �  �    @  �  �  _    n  �  �    6  B  *  �  �  �  a  2    �  �    �  <  �  =  ]  �  �  �  _  5    �  �  B  �  �  Z    �  ]    �  �  �  �  �  �  �  �  �  �  �  �  l  T  5    �  �  �  p  F  !   �   �            !  $               �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  [  A  $    �  �  �  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  �  �  {  g  S  =  %    �  �  �  �  �  g  >    �  y    �  �    %  :  !    �  �  }  F    �  �  g  8    %    �  �  <  2  )         �  �  �  �  v  Y  ;     �   �   �   �   �   �  7  4  2  /  +  '  #      �  �  �  �  �  `  3     �   �   _    #      �  �  �  �  �  �  b  -  �  �  �  \    �  �    ?  t  �  �             �  �  X    �  �  l    �  �  �  �  �  �  z  v  _  F  1          �  �  �  �  8  �  {  �  (          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  {  o  e  b  [  K  >  M  v  n  T  (  �  �  Z    �  V    �  �  �  �  �  w  h  Y  J  :  '    �  �  �  �  �  }  e  M  ^  P  C  6  #    �  �  �  �  ^  5      �  �  �  ]  5  %  C  M  V  _  f  k  n  o  n  h  \  K  .    �  �  �  l  1   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  o  k  s  {  �  �  �  �  �  �  �  �  �  �  �  �  �  o  T  9      �  �  �  �  �  �  {  0  �    �  �  �  �  �  �  �  �  �  �  �  �  |  k  X  E  1      	-  	  �  �  �  b  %  �  �  E  �  �  T  �  J  �  Z  �  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  r  p  n  m  b  c  a  \  S  G  8  '    �  �  �  l  :  �  �  W  
  �  �  �  �  �  �  �  �  �  �  �  �  ]  .  �  �  �  k  =      '  Z  T  N  F  =  3  (      �  �  �  R    �  �  .  �  a   �  �  �  �  �  �  �  y  j  [  N  C  .    �  �  �  �  �  �  �  x  v  g  O  +     �  �  w  N  ,    �  �  �  m  F    �  *  �  �  �  �  �  �  �  �  �  �  �  e  3  �  �  M  �  |  A    #       �  �  �  ~  P  !  �  �  �  _  -  �  �  �  X     �  �  �  �  �  t  0  �  �  D  �  �  x  7  �  �  �  �  �  H        !      �  �  �  �  q  K  #  �  �  �  Q     �   �   H  �  �  ~  g  Q  R  Y  C    �  �  �  �  f  ?    �  �  �  s  C  -    �  �  �  t  A  
  �  �  W    �  �  Y  *  �  �  ?  (  g  �  X  &  �  �  h    �  f    �  O  �  o  �  e  �  7  �  �  |  q  g  [  L  =  *      �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  q  ^  L  �  �  ,  u  v  r  n  f  Z  G  .    �  �  �  m  7  �  �  H  �  {    v  n  f  ^  V  M  B  7  ,  !      �  �  �  �  �  l  H  %        
�  
�  
�  
Z  
  	�  	^  �  �  &  �  -  �  �      W