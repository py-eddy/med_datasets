CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��b   max       P�(     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��S�   max       <���     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F������     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v~z�G�     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q@           �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       <�9X     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�O�   max       B4:7     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�vi   max       B4��     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�74   max       C��     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�~   max       C��l     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��b   max       P���     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊r   max       ?��1���.     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       <���     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F������     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v}��R     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q@           �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�]@         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�=�K]�   max       ?�� ѷY        `�         i         /               	   5   "   B         h      '         4         4   G            *               0                  $                  
                        	                                 2                        NY�NQ�P�(N�� N褷PD��N�Z�O.�PN\�FNh�xO$!.P�QhO��P���N��0N���P��NB��P]�O{BO^�PG�Om$�N��OȥCP���OGO3��NݜDO���N4FOD��O��NGD�O�bFN�kTNJk/O3��O=8O4��P,U-N�a�N�	|O(J8O%YN?�O+��M��bNbN+/mN�diN��=N�F�N
ԂNLBNeN�NӷJNn�N	��O5QN2�_OFQOY��OdF�O���OV��OT=	NƺO9P�O	iN�9KO��N�@ M��<���<ě�<�1<�C�<D��<#�
<t�;��
��o�o��o���
���
�ě���`B�o�t��#�
�49X�D���D���D���D���T���e`B��o��t���t����㼴9X��9X��j��j�ě��ě����ͼ�/��/���o�o�o�C��C��C���������w�#�
�''',1�49X�8Q�D���D���D���D���L�ͽT���T���Y��u��+��+��C���O߽�\)�������㽛���S� "	�����������������������#0}������XI0���//<HU]a]UH</*(//////KN[gmt���xtig[VNIFKK)BN[diifY>) 	
#/34/)#
						�����������������������������������
### 
	����������������������������]y������!������za]��������������������;Ta���������zaH?>@9;:<IKUbn{~{vnb[UIE<::"#*/<DHSKH=<;/+#""""������"���������������������������������������������������	"#%(*)'"����#/6<HUbd`\XUH@<#!#AHnz�����������~xd[A��������������������*/6<HJSUZYUH<1/..+)*FMTamz�������yaPE@@F�������$FOC*������)-56BNOTUWXYNJB54+))nu���������������umn[[\hmt~���xth[XRPR[[���������������������������������������� #,/1<HTVUNH?</# z{��������������zzyz��������������&)5BN[gt}����tg[B)!&5<HUUZ[VUNH<83125555��������������������T[_gt���������tgb[WT)6;BJO[_hjidb_[O6) )��������������ENt���������t[NFDECEttw������������~ytst��������������������������

�������IUanqz~����zunic[UMI��������������������)26:61,)!		
$)668664)($$$$$$$$$$stx~�������vtpssssss�����



���������&)157BCN[dd[XNB5-)&&JOR[htvutnh[OLJJJJJJ"#0:<AB<810%#"�������������������5<=ILPQI=<9055555555��������������������������������������������

����������#+0850##IUbnruwuolfb[USOD?@ILNT[`db][WNNLLLLLLLL������������������������������������������������ ����������q{�������������yuspq�������
"!!
��������	##%"���{|����ztsz{{{{{{{{{{��%$ ������tz����������trrooopt-//6<HHRURHG</**----@BCKNW[ghnrpg\ZNFBA@y���������������yyyy�������������������������������ÿĿƿĿ����������������������U�R�H�=�<�7�5�<�C�H�U�V�Y�V�U�U�U�U�U�U���������V�P�A�8�@�Z��������/�.���	��ŭŬťťŦŭŹ������������Źŭŭŭŭŭŭ�Z�T�N�V�Z�^�f�f�g�s�w�������y�s�f�Z�Z���g�Z�A�5�;�5�/�1�A�Z�s������������������������������������������������������������������
�����!���
���H�G�E�H�H�M�U�\�^�\�Z�U�H�H�H�H�H�H�H�H�����������������������������������������4�(�����
���(�4�<�A�M�O�N�I�A�5�4Ƨ�\�>�/�*�B�=�+�L�\ƁƒƧ��������ƵƯƧ�����������������$�0�=�A�C�=�0�$����������տѿ׿ݿ���(�s�����������s�A���ʼƼ����������������ʼ̼Ӽּټּܼμʼ��6�1�)�(�(�)�+�6�B�F�O�P�O�F�B�7�6�6�6�6����л��ûл��4�M�f�r�����������f�@�������������������������������������������"�;�`�m�������ƿпϿ�ݿĿ��y�G�.�&��"�"���	��������	��"�/�;�?�@�>�;�/�"�@�6�4�3�4�7�A�M�Z�f�s�z�}�s�q�h�f�Z�M�@���z�~�x�c�a�g���������	��"�"�	���������L�A�8�4�(�#�4�A�Z�f�s���������s�f�Z�L�
��
��	�
���#�/�4�9�7�<�>�<�/�#��
ĿĳĭĦĤĩĳĿ�������
��
����������Ŀ����d�d�o�����ʼ��.�D�K�N�H�:���ʼ��a�V�J�I�C�G�I�V�b�o�{ǈǋǈǅ�~�{�o�d�a���������������������������ʾ˾ʾɾľ����L�L�L�R�Y�_�e�i�r�w�~�������~�r�e�Y�L�L�>�4�;�<�:�5�?�L�Y�r�����������~�k�e�L�>�������������������������������������������������������������������#�����E�E�E�FFFFFFF$F+F=FBFDF=F1FFE�E�ݿܿԿ׿ܿݿ������ݿݿݿݿݿݿݿݿѿϿ����������������Ŀݿ���������ݿѾ�������	���"�&�%�"��	�������ŠŠŔňŎŔŞŠŭűŹ������ŹŭŠŠŠŠ������������	���������	���ܹعϹ������������ùϹܹ�����������ܾ������ʾ����������ʾ׾�����
��	����������������)�5�A�A�F�L�P�F�5�)���������������{�����������Ľ̽нӽнĽ�����������*�6�C�J�G�C�6�4�*������ìáàÓÒÛàâìùû��������������ùì�m�a�a�j�m�v�z�����������������������z�m�b�]�V�b�o�{Ǆ��{�o�b�b�b�b�b�b�b�b�b�b���~������������������������������������ùôøù������������ùùùùùùùùùù���|�s�g�^�e�g�h�s������������������������޺ֺɺ��������ɺҺֺ�����������<�8�0�(�#������#�,�0�5�<�D�I�K�I�<����ܾؾ޾���������������𽫽��������������������ĽƽнֽнĽ�����ÇÄ�z�q�zÇÓÙÙÓÇÇÇÇÇÇÇÇÇÇ�������(�0�(�&�����������Y�T�M�K�B�M�P�Y�_�f�j�f�f�_�Y�Y�Y�Y�Y�Y���������������
�����$�%�$������������z�x�w�v�z���������������������������û��ûǻлܻ�ܻܻлûûûûûûûûûû������������ûлܻ���������ܻлû��������������������������������������������������������������#��������޿`�G�+�"����� �"�.�;�G�\�m����y�m�`�������������Ŀѿݿ����� ������ݿɿ���ĚčăĆĚĦĳĿ��������������ĿĦĚEiE\EPECE7E*EEE!E4ECEPE\EiEuExE�E~EuEi����ݽнĽ��������ƽнݽ������������������ĿʿɿĿ������������������������������!�.�:�C�S�_�V�S�L�G�:�.�!�º²¬²º¿���������
����
��������ºD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÓÊÇ�z�x�z�|ÆÇÓàìôùùòìáàÓìâàØÙàìù������������ùõìììì��ߺֺӺֺ���������������� 7 e X A = J 9 ? z i $ B  [ } C 2 , 7 H - ` 6 6 4 ] 4 k F ; = Z R T - * � A M ] A I 6 4 R P 9 X � j � < 8 N T Q e r D ? T J j l j S c 4 % � 2 # J j    �  	.      �  �  �  �  �  ^  �  v  -  0  �  O  J  �  w  �  �  �    �  z  P  �    %  J  �  �  �  �    �  �  �  �  6  3  �  g  �  M  �    ^  t  S  �      w  �     �  ,  �  `  �  3  &    �  �  =  �  �  �  1  �   �<�9X<�C����纃o;��
��P;ě���`B�T�����
�T���m�h��㽓t��T�����
��`B��o�H�9��j�o��o���ě���+�� Ž8Q���t���o��/��P�\)��`B��t��,1���8Q�@��P�`��C��@��49X�L�ͽ8Q�49X�D���#�
�,1�<j�D���49X�aG��<j�Y��Y��]/�Y��P�`�e`B�aG���o�����\)���w��h��1���P��Q콬1���`��E���-��l�B��B ��B&�BLB�B$�B��BzeBuB$P�BcB��BrWA��3B'vRB��B��B�yB+M�A�O�B��B�B!k0B��A���B-��B�0B4:7B��B!k�B��B��BJ�B�VB�B�BLtB	�BB�iB��B
&}B� B��B�CB��B+)B�	Bq�B�B#�UB�Bn�B%��B3LB&g�B�B��BsaB%{�B'�HB�0Bi�B�B8RB
��B��B(�B�BB��B
-B�B�B
�B�BMHB �B&�DB:"B�@B�nB��B=KBcGB$g�BK�B�[BV7A�DB'|B��B��B��B+=&A�viB: B�TB!Z=B;�A�~�B.@�B�mB4��B��B!�EB �B�8B?PB�B�B2B>�B	��B��B��B
?�B�nB�+B.,B��BBhB��B�~B��B#�fB�B��B%I�B?�B&��B�-B�BuFB%N�B'AB��B�YB��B�B
�BB��B=BB��B��B
J�B� B�UB?sB7xAvƖAċ:A��A���AA�9A��A.��A�	A�E@�;oA7��BJ�B	_:A��@��mA׮�@�aVA��Aj]A��A=��A��A@��A�?�A�A/Ba�AL�J?��?��@���A�ZnC��A}VkAz1AZ�A�aJAY�&>�74AT/2A�X�A#2,A��IA�E�A��B��A��eA�c�A��_@9qA� AV�BA#
�Aɲ�A4�@�p~B��A�K^@��@���A��A���Ae�Az.�A㫖C��A+��Aw�+A�7A�lC��A��A�'�@F�Av�~AĂ�A�a�A��AA�'A��pA/]�A��|Aś�@��	A7�wB6�B	��A�%�@�%�A׈�@�3A�Ah��A��UA<�A�tA@��A�}qA�p�A��B@XAK��?�z�?��@��A��C��lA}"'Ay�AZ�A�~�AY-)>�~AS�A��dA"NSA���A̞�A��;B��A�}�A΀�A�$�@5?VA�t�AWKA"�!A�$A5�@�B��A���@��M@���A��A�DAc��AylA��C��LA0�Aw CA"�A��C��A��Á@C�         i         0               
   6   "   C         h      '         5         5   G            +               1                  %                                          
   	            	                  3                                 Q         5                  ?      ;         A      6         7            C            %               !                  +                                                                        #                                    ?         '                  #      +         5               /            C            !                                 #                                                                        !                           NY�NQ�P�(�N�� N���PDNN�Z�O.�PN\�FNh�xO$!.PolOP,�P0�DN��0N���P��NNB��OIg�N��UOC��P��N�lRNţWO���P���OA?O3��N�fO��N4FOD��N�8NGD�O���N�H�NJk/O!xO=8O>O�`�N���N�	|O{�O%YN?�O+��M��bNbN+/mN�diN��=N�04N
ԂNLBNeN�NӷJNn�N	��NüZN2�_OFQOY��OdF�O��-OB,OT=	NƺO9P�O	iN�9KO��N�@ M��  �  *  �  �  
  �  �    �  G  �    �  �  _  �  
@  *  �  Y  �  +  4  H  t  �      _  �  7  {  �  �  �  �  �  �  �  �  A  /  p  P  �  y  -  �  [  
    �  p  $  I  �    
  �  b  �  �  �  �  "  �    v  �  "  8  �  >  <���<ě����
<�C�<#�
;o<t�;��
��o�o��o��/�u��9X��`B�o�ě��#�
�+�e`B�e`B���
��t��e`B��/��o���
��t���j��󶼴9X��j�ě��ě��o��`B��/�����\)�#�
�+�C��t��C���������w�#�
�'',1�,1�49X�8Q�D���D���D���L�ͽL�ͽT���T���Y��}󶽍O߽�+��C���O߽�\)�������㽛���S� "	�����������������������0U�����nP<0���//<HU]a]UH</*(//////JNQ[dgtvtpg[NMJJJJJJ)5BNXbeb]TI5)	
#/34/)#
						�����������������������������������
### 
	��������������������������������������������������������������������GJLVz���������zaTJHG:<IKUbn{~{vnb[UIE<::"#*/<DHSKH=<;/+#""""����������������������������������������������������������	"#%''#"	���#%/<HU_b^ZUHD</$#Uan������������~p`UU��������������������///<HRUYXUH<//,+////PTZamz}������zaZNKJP�������$FOC*������.59BMNSTVVNB55,*....nu���������������umnV[hity���tnh[ZTRVVVV���������������������������������������� #,/1<HTVUNH?</# ~���������������~|~~��������������17DN[gty����~tg[B318<HOUWYUSJH<;5348888��������������������W[cgt��������tgf[ZWW)6;BJO[_hjidb_[O6) )����
	���������LTgt��������tg[VOMMLstux�����������zutss�������������������������

���������IUanqz~����zunic[UMI��������������������)26:61,)!		
$)668664)($$$$$$$$$$stx~�������vtpssssss�����



���������&)157BCN[dd[XNB5-)&&JOR[htvutnh[OLJJJJJJ##08<@A<70.#####�������������������5<=ILPQI=<9055555555��������������������������������������������

����������#+0850##FIUZbnqrpnicb^UKIFFFLNT[`db][WNNLLLLLLLL������������������������������������������������ ����������t�������������{wuurt�������
!
��������	##%"���{|����ztsz{{{{{{{{{{��%$ ������tz����������trrooopt-//6<HHRURHG</**----@BCKNW[ghnrpg\ZNFBA@y���������������yyyy�������������������������������ÿĿƿĿ����������������������U�R�H�=�<�7�5�<�C�H�U�V�Y�V�U�U�U�U�U�U���������t�`�Y�[�s������������
������ŭŬťťŦŭŹ������������Źŭŭŭŭŭŭ�f�]�Z�S�Z�\�e�f�j�s�z�~�t�s�f�f�f�f�f�f�����f�H�D�A�:�A�Z�s����������������������������������������������������������������������
�����!���
���H�G�E�H�H�M�U�\�^�\�Z�U�H�H�H�H�H�H�H�H�����������������������������������������4�(�����
���(�4�<�A�M�O�N�I�A�5�4�h�L�G�O�W�[�]�c�uƁƎƩƶƼƺƮƧƎ�u�h��	��������� ���$�%�0�:�=�=�8�0�$���g�N�(���������(�A�g���������������g�ʼƼ����������������ʼ̼Ӽּټּܼμʼ��6�1�)�(�(�)�+�6�B�F�O�P�O�F�B�7�6�6�6�6���������4�@�Y�r����������M�@������������������������������������������m�`�T�L�=�;�8�;�>�G�T�`�y�����������y�m�/�)�"���	���	��"�/�;�<�=�<�;�5�/�/�M�C�A�9�5�5�9�A�M�Z�f�t�x�s�n�j�f�f�Z�M�������f�g�s��������������������������Z�P�M�B�F�M�Z�b�f�s�v�����}�t�s�f�Z�Z�
�
�
��
���#�/�3�7�5�6�/�#��
�
�
�
ĿĹĳĮĭİĳĿ��������������������Ŀ����d�d�o�����ʼ��.�D�K�N�H�:���ʼ��V�K�I�D�I�J�V�b�o�{ǈǃ�|�{�o�b�V�V�V�V���������������������������ʾ˾ʾɾľ����Y�R�W�Y�c�e�q�r�r�|�~����~�r�e�Y�Y�Y�Y�@�=�?�A�=�I�Y�e�r�~�����������~�r�Y�L�@�������������������������������������������������������������������#�����FFFFFFFF$F'F1F=FAFCF=F9F1F$FFF�ݿܿԿ׿ܿݿ������ݿݿݿݿݿݿݿݿѿĿ��������������Ŀѿݿ��������ݿѾ��������	���"�#�#�"��	�������ŠŠŔňŎŔŞŠŭűŹ������ŹŭŠŠŠŠ��������������	�������
�	�����ܹعϹ������������ùϹܹ�����������ܾʾȾ��������ʾ׾����� ������׾ʾ����������������)�2�8�=�B�A�;�5�)������������������������ĽʽнѽнĽ�������������*�6�C�J�G�C�6�4�*������ìåàÕÞàäìù����������������ùìì�m�a�a�j�m�v�z�����������������������z�m�b�]�V�b�o�{Ǆ��{�o�b�b�b�b�b�b�b�b�b�b���~������������������������������������ùôøù������������ùùùùùùùùùù���|�s�g�^�e�g�h�s������������������������޺ֺɺ��������ɺҺֺ�����������<�8�0�(�#������#�,�0�5�<�D�I�K�I�<����ܾؾ޾���������������𽒽����������������ýĽνĽ�������������ÇÄ�z�q�zÇÓÙÙÓÇÇÇÇÇÇÇÇÇÇ�������(�0�(�&�����������Y�T�M�K�B�M�P�Y�_�f�j�f�f�_�Y�Y�Y�Y�Y�Y���������������
�����$�%�$������������z�x�w�v�z���������������������������û��ûǻлܻ�ܻܻлûûûûûûûûûûû������������ûлӻܻ����ֻܻлû��������������������������������������������������������������#��������޿`�G�+�"����� �"�.�;�G�\�m����y�m�`�������������Ŀѿݿ����� ������ݿɿ���ėĉċĚĦĳĿ���������
��������ĿĦėEiE\EPECE7E*EE#E*E6ECEPEiErEuEwE�EzEuEi����ݽнĽ��������ƽнݽ������������������ĿʿɿĿ������������������������������!�.�:�C�S�_�V�S�L�G�:�.�!�º²¬²º¿���������
����
��������ºD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÓÊÇ�z�x�z�|ÆÇÓàìôùùòìáàÓìâàØÙàìù������������ùõìììì��ߺֺӺֺ���������������� 7 e Y A 0 M 9 ? z i $   Y } C ( , K 3 / X ( 9 . ] + k E : = Z 6 T  * � < M R * L 6 - R P 9 X � j � < 3 N T Q e r D 0 T J j l l K c 4 % � 2 # J j    �  2    �  �  �  �  �  �  ^  1  �  f  0  �  D  J  �    �      �    z    �  �  �  J  �  "  �  o  �  �  H  �  7     �  �  4  �  M  �    ^  t  S  �  �    w  �     �  ,  �  `  �  3  &  �  �  �  =  �  �  �  1  �   �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  �  �  �  �  ~  t  j  `  U  K  @  6  +  !    *  2  :  A  G  L  H  >  4  %      �  �  �  �  �  g  I  ,  �  h  �  L  �  �  �  r  -  �    z  �  `    �  :  �  �  D  �  �  �  �  �  �  �  �  |  d  F    �  �  C  �  �  j  @    �  �        	        �  �  �  �  �  �  �  �  �  x  e  #  �  �  �  �  �  �  j  9  �  �    E  �  �      "  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  s  K  #  �  �  �  �  d  9    �  �  �  �  G  @  9  2  +  $        �  �  �  �  �  �  �  ~  i  U  @  �  �  �  �  x  j  [  I  6  "    �  �  �  �  �  �  �  �  �  �    R  �  �  �  �    
    �  �  �  �  u  :  �  '  K  e  t  �  �  �  �  �  �  �  �  �  �  k  F    �  |  	  s  �    �  3  �  �  �  �  �  �  �  u  N  "  �  �  �  v    ^  �  �  _  W  P  H  ;  ,            �  �  �  �  ~  `  C  %    �  �  �  �  p  Z  C  '    �  �  �  S    �  �  �  d  E  (  	\  
  
9  
@  
1  
  	�  	�  	~  	'  �  )  �  �    P  �  Z  �  �  *         �  �  �  �  �  �  m  T  ;  #    �  �  �  �  �  �  �  	  
        /  U  s    �  |  X    �  c  8  �   �  C  O  X  Y  W  S  G  9  $    �  �  �  �  k  C    �  �    �  �  �  �  �  �  x  _  @    �  �  �  U    �  �  Y    �  �      *    �  �  �  ?  �  �  S      �  n  �  Q  o  �  �  �      *  2  4  2  +  !      �  �  �  }  J    �  �  F  G  G  F  F  F  B  <  -      �  �  �  �  �  Q    �  �    4  U  l  q  s  n  e  M    �  q    �  >  �  .  Q    �  �  �  �  z  J    �  �  �  �  �  s  >  �  �    �  �      �      �  �  �  �  �  o  H    �  �  G  �  g  �  (     �        �  �  �  �  �  �  �  �  �  �  �  �  i  8  �  �  U  W  Z  [  ]  ^  X  H  /    �  �  �  b  5    �  �  �  v    (  T  k  ~  �  �  i  O  ;  ,    �  �  �  2  �  �  )  �  �  7  4  2  /  ,  %          �  �  �  �  �  �  �  �  �  �  {  n  ]  J  5       �  �  �    U  )  �  �  �  I  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  Z  C    �  �  �  �  �  �  �  �  �  �  �  z  h  V  E  3       �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  )  �  �  J  �  �  5  �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  g  ?    �  �  �  J  �  P  �  �  �  �  �  �  �  �  �  �  �  w  k  _  Q  ?  -      �  �  �  �  �  �  �  �  �  �  �    X  ,  �  �  �  M    �  �  �  �  �  �  �  \    �  �  �  �  �  ~  `  >  /  '    �  �  |  �  �  �  �  �  �  �  �  �  h  >    �  �  �  �  \  +  �  �  (  (  <  A  A  <  .       �  �  �  b    �  `  �  t  �  �    &  )  !         �  �  �  �  �  �  �  k  S  @  1    �  p  h  _  V  K  ?  1  !      �  �  �  �  �  �  �  v  f  V  E  J  O  P  O  K  C  7  '    �  �  �  �  q  ?  �  �  _  �  �  �  �  �  �  �  �  �  z  j  Y  G  4    	  �  �  �  f  )  y  n  c  X  C  -    �  �  �  �  |  Z  7    �  �  �  z  T  -        �  �  �  �  �  �  e  H  $  �  �  �  v  F   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
    !  ,  [  Q  G  =  3  )      �  �  �  �  �  �  r  T  7     �   �  
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  j  \  P  C  2  !    �  �  �  S  
  �  ~  s  i  _  U  J  A  8  /  %         �   �   �   �   �   �  W  i  o  l  i  a  X  M  A  2       �  �  �  �  m  K  )    $        �  �  �  �  �  �  �  �  �  �  �  �  w  l  a  V  I  B  :  +    
  �         �  �  �  �  �  �  S  #  �  �  �  �  �  �  �  �  �  �  n  V  ;       �  �  �  }  [  7      �  �  �  �  �  �  i  M  0      �  �  �  �  �  �  s  Q  
  �  �  �  �  �  �  �  �  b  C  $    �  �  �  q  J  #   �  �  �  �    |  x  t  q  n  k  i  f  c  `  ]  Z  W  S  P  M  N  T  Z  ^  `  b  a  `  `  `  ^  \  Y  R  L  <  )     �   �  �  �  �  �  �  �  �  �  x  a  I  1    �  �  �  �  q  Q  1  �  �  �  �  �  u  d  M  3    �  �  �    �  �  W  -  �  �  �  �  �  �  �  d  :    �  �  �  o  L  *    �  �  |  K    �  �  �  �  i  g  k  G  "     �  �  �  �  �  m  J    �  W      !      �  �  �  �  i  K  .      �  �  �  ;  �   �  Z  �  u  K    
�  
�  
g  
'  	�  	�  	-  	  �  .  x  �  �  �  �        �  �  
    �  �  �  �  �  u  G    �    "  �  Z  v  `  J  3    �  �  �  �  �  w  _  H  =  N  ^  k  i  h  f  �  �  �  �  �  �  �  �  y  Z  :    �  �  �  �  �  �  �  �  "          �  �  �  �  �  �  �  x  Z    �  g  #  �  �  8  #  
  �  �  �  �  \  (  �  �  >  �  R  �  ?  �  �    7  �  �  �  �  �  i  P  1    �  �  �  �  v  K    �  �  �  B  >    �  �  �  �  _  7    �  �  �  �  n  L  '    �  �  �    �  �  �  �  �  �  �  �  �  �  ~  q  d  W  J  =  0  "  