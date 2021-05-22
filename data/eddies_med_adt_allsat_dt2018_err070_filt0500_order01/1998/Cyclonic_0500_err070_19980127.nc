CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����+       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�l   max       P��]       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @F}p��
>     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vf�G�{     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @4         max       @N�           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�``           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       <o       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�*�   max       B3?&       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��'   max       B3>�       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��p   max       C��       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >S��   max       C�'       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ]       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�l   max       P��j       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?�>BZ�c        AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <�j       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?O\(�   max       @F}p��
>     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vf�G�{     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @O            �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�6��C-   max       ?�"h	ԕ     @  [L                           ]   B         3               3      :   @   <            -         U   E                     6               #      /      5      
                        !      	               
               O	��N�+N@��N���NC��N�M�lN���P��PP��]OE�N�\�P	��OǟO7�NF�oN�xZP �nOmbPPt�GP���O�']N��`O��N��O��jNI��NJ*�P���O�o�NGx�O�N�N�6�O���ND��O�(N�*YNڨ�N0��O_2:O��N�56Ol�{Nh�2O�ؽOk�N���N��OaΟN_C�OE�qN���OS}O3�9OlM�N�>�N���NBN�`�N>��O6�fN�3vN
�qOD��N�COM�}eN��<���<���<e`B;ě�;o��o�D���ě��t��t��#�
�49X�49X�D����o��C���t����㼛�㼣�
���
���
��j��j��j�ě��ě�����������/���������o�o�t��t���P���,1�,1�,1�0 Ž49X�<j�@��D���D���H�9�T���Y��]/�aG��aG��aG��u��%��+��+��C���\)���P���
������Ƨ�")66>>:6)"����������������������

�������������������������|����&07<HFIII<0+&&&&&&&&FHPUY^_`abbaUHHEACBF����

������������������������������z���������
������ywz�#0<Ip��������n<#�����������������������),)!�����
/<JZeaUN<#�����������

�������� ������HHMRUafea^UUSHHHHHHH��������������������")5Nt��������tg[B0!"�������������;Taz�����~zmaTNB226;ls���������������zolK[ht������������hOJK#/<HOMJHC</,#����������������������������������������#/<HSZ[Y\aUH/#ghlt����xtihgggggggg��������������������y���������������xssyTU_z�����������zna[Tqu�������xurqqqqqqqq1=BLNS[\]\[VNHB@8511X[htvtnh[VXXXXXXXXXXY[fhhtxzthf[TVYYYYYY�)6BMPR[VOB)�����������������������������������������T[\ahjt�����{th[TTTT���
"
�����������������������������mz�������������ztlim������������������������� �������������*6COUYYSOC6*"������������������������������������yz�����������}zwutuy)59855.)egpt��������~tggeeee�����������������������������������������������������������,/<GHSJH<<;/-+,,,,,,�������������������������������������� 0<CEC<:0#
 z�����������zpzzzzz������������mnz{���zonfmmmmmmmm8<@IU\bdggb`USIB<888����������������������������������������KNT[gjhge[[NHHKKKKKKJN[b`[ONNNJJJJJJJJJJ��	)**#"����� ���      "#&/14/###""""""""""559ABN[]^\[XUNB=6555�ɺ����������ɺʺֺ�����������ֺպɽĽ����Ľнݽ����ݽнĽĽĽĽĽĽĽ�ŔŉŇŔśŠŭŮŰŭŠřŔŔŔŔŔŔŔŔ����������������������������������޼�޼ּѼּ��������������������������������������������������Ѽf�b�Y�R�M�I�M�Y�^�f�i�i�f�f�f�f�f�f�f�f�/�*�#�"�#�/�<�H�U�^�W�U�H�<�/�/�/�/�/�/�������ôëó���)�BāĚħĦā�k�[�6�������r�S�@�8�(��(�A�Z�����������������U�a�n�q�p�r�n�n�h�a�`�U�T�H�H�A�=�A�H�U�O�G�C�6�*�#����$�*�.�6�C�E�O�U�[�O�O��ùÞÞáíÿ�����������������������ž���׾ʾþ������ʾ׾ܾ����������𾺾������������ʾ׾������������׾ʾ����	��������	���"�#�"��������f�`�Z�T�S�R�Z�f�s�y���������s�f�f�f�f�	��������	��.�D�G�T�M�I�A�5�.�"�	àÓÃ�o�n�r�t�zÇÓßàæìöÿüùìà�s�B�5�3�7�N�s���������������������sƳƚƁ�u�O�;�O�\�uƎƧ������(�%�����Ƴ�@�-�"�%�4�;�Q�Y�f�������������b�Y�M�@�)�"� �$�$�)�6�B�E�O�Y�Q�O�H�B�6�)�)�)�)�r�i�e�d�d�e�l�r�~�������������������~�r���������ƺɺ˺ҺӺɺ��������������������M�A�:�8�8�;�A�M�f���������������s�f�M����¿³³¿�����������������������������������(�*�)�(�����������r�L�3�3�>�T�c�������ɻ�����ֺ������r�Ϲù����������ùܹ������!� ��	����Ͽ;�6�5�3�;�G�T�V�W�T�N�G�;�;�;�;�;�;�;�;��ݿڿѿƿѿ������� ����������ù������ùŹϹ׹۹Ϲùùùùùùùùùú���	����'�2�2�3�>�3�'�������������w�q�^�\�x�������������������������ʼǼʼμּݼ�����ּʼʼʼʼʼʼʼ����������)�B�O�[�f�h�q�j�[�B�6�)���.�#�"���	����	��"�"�.�2�1�5�.�.�.�.D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������ĿĸĸĻĿ�����������	�
��
����������Ŀ���� �,�<�U�~ŇŠŹŽŲŠŇ�b�I�0�#��;�4�3�;�;�H�T�[�`�[�T�H�;�;�;�;�;�;�;�;�����ܾ׾پ����	����"�#�"�!��	���/�&�/�/�7�;�A�H�N�L�J�H�@�;�/�/�/�/�/�/�������'�4�@�Y�n��������f�Y�4��������������������������������������������������u�t�������������������������������������������������������������������������������������������������	���$��������������������������㿸���Ŀѿؿݿ������������ݿѿĿ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��g�]�N�G�A�>�F�J�N�g�s���������������w�g�нͽŽ½��Ľнѽݽ����	�������ݽн�������������������ĽѽֽֽнʽĽ������������������ĿѿտѿϿɿĿ����������������v�x�����������������������������������ûûûûͻлӻܻ޻ܻܻлûûûûûûûûû������������ûлܻ�����ܻллû����	�����!�"������������U�I�<�0�#������#�0�D�U�Y�b�n�u�s�U��������������� �
�����
������������¦¦²º¿¿¿²¦¦¦¦¦¦¦¦¦¦�S�L�:�,�$�+�.�:�G�S�b�v�y�������y�l�`�S�������������ĽнܽԽнĽ���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�_�U�H�G�G�H�H�U�a�e�n�zÃÁ�z�n�g�a�a = 5 M 1 K Z o I U U T p ` f V Y 8 D a ` > q 4 ) < , ; k A ( : B P 9 K M H ] J ^ = x + 3 z G W P j U 1 [ / 1 : + r f b . r k Y I d a y 0  4  �  a  �  w  ,  H  �  {  [  t  �  �  �  �  o  
  �    �  �  �  $  ,  H  �  [  �  2  �  n  O  B  �  I  |  c  �    R  �  �  �  �  �  �  g  �  �  �  t  �  �  �  �  �  �  �  Z  �  �  �  �  ,  �  �  4  �;ě�<o;�o�o�o��o���
���
������P��h��o�y�#������h��j�o��O߽\)���-��1���
�<j�\)��/��O߼�`B����G��ě��C��0 Ž�P�,1�ixս�㽸Q�'Y��0 Ž�C�����aG���E��@��ȴ9��O߽m�h�e`B��o�e`B��%��hs��7L���
��-��+��hs���㽓t���\)��1���罩��\��-�ȴ9��`BB�B)�B��BV^B&�B�aB#��B,�B^�B&�B!�B�kB��B��BVOB�B uB	#,BA�*�B��B��Bj�B!��B!.�B��BR�BCBe\BWfB3?&BRB�LB��B��B �>B�gB�B}mB5�B i�Bm�BAB0"uB��B��BǯB�1B
38B
tB,�7B� B��BR�B"z�B%\xB�B1wBB�B'�BsyBE�B��B�jB9B�B��BB?�B)��B�jBE�B&1�BșB#�[BF�B�B'�zB!"KB�_B�B��BC&B�jB @�B	��B#iA��'B?XB�]B?�B!�.B! uB@�BM,BO�B�B@B3>�B?gB��BăB<�B B[B��B53B@KB<.B |RBC�B?:B05�B�AB6�B�BőB	��BԎB,��B�DB��B:�B"?�B%�JB>�B3�B0UB'NB#�B?WB��B��BAgB?5B;�Bý@@�A*.A�'pA��Al$A�^@�A��]A�9	A��A�B d%A�5�AT@<AR��A[c�AB�A\~�AʽMA�l�B�H@��JA׈�@F@0@)A@�A��A�0�@A]>޿�AdQ�A�R>��p?�1 @�)�A��A�FA^$vC��A�-�A��qA��A���AY�TA�T�@ӘA�p�A��lAr��A�%ALCA}7C��A��@A,0TA"�Ax�A���@���@���B	" A��wA���A�2�A�A%1C��Aƌ @C�A+dA�[A�~�A�1A�Y@��A��~A׎�A���Aŀ�B �@A�(AS�AR��A[T�AB�%A\{�AʛYA��zBA@���A׀@�5@,�4A?�A�NA�|f@G{>�+�AehA�x�>S��?���@�'�A ��Aց�A^(6C��A�]�A�`�A��A���AY�A��@�0A��ZA��Ar��A��0AD�A|�C��A��A,��A"QAx�gA�f@��@�	�B	@A���A燏A�YA�A%	�C�'AƠ                           ]   B         3               3      :   A   =            .         U   F                     7               #      /      5         	                     !      	               
                                          C   C         -               '      3   ;   -                     9   #                                    )            '                                                                                             +   ?                        #      -   5                        5                                       %            '                                                                  N�!JN�+N@��N��2NC��N��QM�lN���P$�^P��jOE�N�\�O���OǟN�[7NF�oN�GOҩ�OmbPP&�6P���OvD�N��`O��N��OJ́NI��NJ*�Pn=O���NGx�O�N�N#��O���ND��Oa..N�*YNڨ�N0��OS۽O���N�56O;�fNh�2O��ZOk�N���N��O"x�N_C�O�[N���O#K�O&�~O<pN�>�Nc�kNBN���N>��O6�fN�3vN
�qOD��N�COM�}eN��      �    	  @    �  �  �  L  �  @  ;  �  m  �  ,  �  4  8  �  �    �  �  �    	�  	�  )  �  �    &     	y  �  /  [  �  �  �  N  �  j  �  �  �       j  �  �    �  {  �  Y  �  �  H  �  T  �  ~  �  &<�j<���<e`B;�o;o�o�D���ě��C��#�
�#�
�49X��j�D����1��C���1��󶼛��C��ě���P��j��j��j��P�ě������o�0 ż������\)�o�o�8Q�t���P���0 Ž49X�,1�L�ͽ49X�@��@��D���D���T���T���aG��]/�ixսe`B�q���u��o��+��7L��C���\)���P���
������Ƨ�)26;:761)%����������������������

������������������������������&07<HFIII<0+&&&&&&&&GHSUV]^_a`UHFADCGGGG����

���������������������������������������������������
#0<Ir������n<#������������������������),)!������
$/<G;/,#������������

�����������������HHMRUafea^UUSHHHHHHH��������������������-<N[t�������tg[N5-)-�������������;@ETamz�����zmaTN@;;v���������������zonvY[ht�����������th[WY#/<HOMJHC</,#����������������������������������������#/<EHQUSPJH</*#ghlt����xtihgggggggg��������������������z���������������zuuz^kz���������zqnf`\[^qu�������xurqqqqqqqq1=BLNS[\]\[VNHB@8511X[htvtnh[VXXXXXXXXXXX[[htuxth^[ZXXXXXXXX�)6BMPR[VOB)�����������������������������������������T[\ahjt�����{th[TTTT���
"
�����������������������������mz������������{zuljm������������������������� �������������%*6CJQUTOLC6*������������������������ ������������yz�����������}zwutuy)59855.)egpt��������~tggeeee������������������������������������������������������������,/<GHSJH<<;/-+,,,,,,��������������������������������������
#07<ACA<70#

z�����������zpzzzzz����	������������mnz{���zonfmmmmmmmm=IPUWabcebUIE>======����������������������������������������KNT[gjhge[[NHHKKKKKKJN[b`[ONNNJJJJJJJJJJ��	)**#"����� ���      "#&/14/###""""""""""559ABN[]^\[XUNB=6555�������úɺκֺ������ �������ֺɺ����Ľ����Ľнݽ����ݽнĽĽĽĽĽĽĽ�ŔŉŇŔśŠŭŮŰŭŠřŔŔŔŔŔŔŔŔ��������������������������������������޼ּѼּ����������������������������������������������������Ѽf�b�Y�R�M�I�M�Y�^�f�i�i�f�f�f�f�f�f�f�f�/�*�#�"�#�/�<�H�U�^�W�U�H�<�/�/�/�/�/�/������������B�h�t�|�g�j�j�g�_�B�6���������s�T�@�8�(��1�Z�����������������U�a�n�q�p�r�n�n�h�a�`�U�T�H�H�A�=�A�H�U�O�G�C�6�*�#����$�*�.�6�C�E�O�U�[�O�Oùìæååèìö������������� ��������ù����׾ʾþ������ʾ׾ܾ�����������ʾɾ��������ʾҾ׾������׾ʾʾʾʿ��	��������	���"�#�"��������f�c�Z�W�W�Z�b�f�r�s�������|�s�f�f�f�f�	����������	��.�5�;�=�:�5�2�'�"��	àÓÃ�o�n�r�t�zÇÓßàæìöÿüùìà�����s�V�G�I�R�g�s��������� � ����������ƚƆ��Q�\�b�uƁƎƳ������$�"�����Ƴƚ�@�>�5�6�<�C�M�f������������r�f�Y�Q�M�@�)�"� �$�$�)�6�B�E�O�Y�Q�O�H�B�6�)�)�)�)�r�i�e�d�d�e�l�r�~�������������������~�r���������ƺɺ˺ҺӺɺ��������������������M�A�?�>�A�B�M�Z�f�s���������{�s�f�Z�M����¿³³¿�����������������������������������(�*�)�(�����������v�L�B�?�E�X�i�����ɺ�
������ֺ������v�ù��������ùϹܹ������������ܹϹÿ;�6�5�3�;�G�T�V�W�T�N�G�;�;�;�;�;�;�;�;��ݿڿѿƿѿ������� ����������ù������ùŹϹ׹۹Ϲùùùùùùùùùú'������'�-�.�3�6�3�'�'�'�'�'�'�'�'�������w�q�^�\�x�������������������������ʼǼʼμּݼ�����ּʼʼʼʼʼʼʼ�������
���)�6�B�O�Z�a�[�B�?�6�)��.�#�"���	����	��"�"�.�2�1�5�.�.�.�.D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������ĿĹĹļĿ��������������
������������Ŀ�<�#��"�-�<�U�nŁŇŠŸůŠŔŇŀ�b�I�<�;�4�3�;�;�H�T�[�`�[�T�H�;�;�;�;�;�;�;�;�����۾ݾ����	��������	�����/�&�/�/�7�;�A�H�N�L�J�H�@�;�/�/�/�/�/�/�������'�4�@�Y�m��������Y�@�4��������������������������������������������������u�t���������������������������������������������������������������������������������������������	�������	�������������������������Ŀ����ſοѿݿ�������� �����ݿѿǿ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��f�Z�N�L�E�N�O�N�Z�g�t���������������t�f�нνƽĽý½Ľнݽ�����������ݽн������������������������ĽͽѽѽнĽ����������������ĿѿտѿϿɿĿ����������������x�y�����������������������������������ûûûûͻлӻܻ޻ܻܻлûûûûûûûûû��������ûʻлܻ���ܻлûûûûû����	�����!�"������������U�I�<�0�#������#�0�D�U�Y�b�n�u�s�U��������������� �
�����
������������¦¦²º¿¿¿²¦¦¦¦¦¦¦¦¦¦�S�L�:�,�$�+�.�:�G�S�b�v�y�������y�l�`�S�������������ĽнܽԽнĽ���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�_�U�H�G�G�H�H�U�a�e�n�zÃÁ�z�n�g�a�a : 5 M + K Z o I 6 T T p 8 f R Y , O a Z : | 4 ) <  ; k : # : B P @ K M ? ] J ^ 6 w + 0 z H W P j 6 1 U / " 4 $ r S b 1 r k Y I d a y 0    �  a  �  w    H  �  �  T  t  �  �  �  �  o  �      G  +  ~  $  ,  H  �  [  �  �  �  n  O  B  M  I  |  �  �    R  �  &  �  �  �  q  g  �  �  i  t  K  �  ^  k  �  �    Z  �  �  �  �  ,  �  �  4  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �       �  �  �  �  �  �  T  #  �  �  S  �  R  �     �    
    �  �  �  �  �  �  �  �  �  �  �  �  �  R    �  �  �  �  �  �  �  k  J  &     �  �  �  ]  2    �  �  w  G    �  �        �  �  �  �  �  w  J  )  �  �  S  �  �  B   �  	    �  �  �  �  �  �  �  �  �  �  �  �  t  m  h  c  ]  X  :  ?  ;  /    
  �  �  �  �  �  �  t  g  Z  K  =  .    �                        
    �  �  �  �  �  �  �  �  �  �  �  �  �  |  Q    �  �  |  B  	  �  �  V    �  �  )      >  |  �  �  p  �  K  �  �  �  k  .  �  �  �    8  �  �  �  e  ;    �  �  d    �  �  q  6  �  �    �  �   �  L  C  *    �  �  �  �  V  *  �  �  �  Q    �  -  ~  �  9  �  �  �  �  �  �  u  g  Y  J  ;  -  +  3  ;  C  9  *      �  �  	    +  ;  %    �  �  z  C    �  i    �  6  �    ;  :  6  -        �  �  �  �  �  �  u  H    �  �  �  �  R  \  c  h  n  u  }  �  �  y  f  >  
  �  �  :  �  �  J   �  m  ^  P  B  0      �  �  �  �  �  �  m  R  8  &  >  W  p  �  �  �  �  �  �  �  �  �  b  B    �  �  �  �  h  J  X  �  [  �  �    %  ,  #          	  �  �  e  �  s  �  o    �  �  �  �  �  �  �  �  �  �  ^  6    �  �  �  �  R  �  �  �      '  2  4  '  	  �  �  _    �  K  �  g    �  �  �    5  4    �  �  �  �  �  �  �  �  �  �  �  Y  �    `  �  *  |  �  �  �  �  �  �  �  �  �  �  s  �  �  g  �    s  u  �  �  �  �  �  �  �  �  �  m  A    �  o    �  [  �  �  (    z  s  k  _  R  D  4  "    �  �  �  �  �  p  K    �  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �         &  3  @  �  K  k  �  �  �  �  �  h  >    �  �  �  7  �  =  �  �  �  �  v  k  `  U  I  :  ,      �  �  �  �  �  �  }  ^  @  "      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	�  	�  	�  	�  	X  	  �  |  T  	  �  U  �  9  �    m  �  �  q  �  	  	^  	  	�  	�  	�  	j  	A  �  �  j    �  E  �  �  �    �  )      �  �  �  �  �  �  �  �  �  �  ~  p  a  Q  B  2  #  �  �  �  �  �  �  �  z  _  E  *    �  �  �  }  D  �  _  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �    w  j  <    �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  u  &  �  �  �  �  �  �  u  S  0    �  �  �  t  �  X    �  1     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      )  4  �  	P  	l  	x  	w  	j  	H  	  �  �  q  !  �  J  �  "  N  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  h  U  C  2  !    /    �  �  �  �  y  S  )  �  �  �  e  1  �  �  �  s  D    [  O  D  9  -      �  �  �  �  n  H  "  �  �  �  ~  R  %  �  �  �  o  H    �  �  �  I    �  q    �  _  �  �  ?    �  �  �  �  �    Z  4      *     �  }  '  �  �  �  �  �  �  �  �  �  �  w  b  M  8  !    �  �  �  z  F  �  �  S  �  H  H  J  N  M  I  A  2    �  �  �  Y    �  ;  �    V  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  i  j  e  F    �  �  z  %  �  @  �  E  �  T  �  Y  �    �  p  �  �  o  Z  A  &    �  �  �  ]    �  �  /  �  R  �  >   �  �  �  �  �  �  �  x  b  K  4      �  �  �  B  �  Y  �   �  �  �  �  �  t  `  I  2    �  �  �  �  �    l  Z  G  3       	            �  �  �  �  j  <    �  �  �  R  �                  �  �  �  �  �  �  �  �  �  q  ^  J  7  K  L  N  ^  i  b  Y  H  5      �  �  �  �  �  �  V  "  �  �  �  �  �  �  g  P  1    �  �  o  0  �  �  ^    �  o    �  �  �  �  �  �  �  �  i  M  /    �  �  �  �  `  &  �  �          �  �  �  �  �  �  _  6    �  ~  &  �  a  �  �  �  �  �  �  �  �  �  �  �  �  p  L  "  �  �  `    �  A  �  {  g  R  =  &    �  �  �  �  �  �  �  �  o  ^  <  �  L   �  S  |  �  �  �  �  �  {  i  W  D  2      �  �  �  r  /  �  Y  P  H  Q  S  =  "  �  �  �  �  P    �  �  x  4  �  �  M  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  H  ;  '    �  �  �  �  �  �  q  V  1    �  �  G  �  �  U  �  �  �  �  �  �  �  o  Q  4    �  �  �    �  {  A     �  T  S  S  R  Q  P  O  M  I  F  B  ?  ;  1      �  �  �  �  �  �  �  \  9    �  �  �  �  �  �  ^  :    �  �      �  ~  }  }  |  {  v  q  l  d  W  J  =    �  �  {  c  R  @  /  �  �  �  �  �  �  �  �  �  |  b  H  .    �  �  �  �  �  m  &      �  �  �  �  �  �  f  =    �  �  �  �  �  �  l  H