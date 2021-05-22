CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��$�/      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       <�/      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @Fk��Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v{33334     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @Q�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @���          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��
=   max       <�1      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0�V      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�g�   max       B0F�      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >-   max       C��O      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >2   max       C���      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          U      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P���      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�0��(�   max       ?���n/      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       <�/      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @FU\(�     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @v{33334     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @���          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��	� �   max       ?���n/     `  U�                                     
         ;   
      
               T   ;                        !      !   E   1               5      !   #   /            '               ,         "               Nh��N�O8U�O/�)N�N��PQ
�O4��N�O:a�O�N�fN��N|�P��O
R�N�S�O��$OP�
OG��N�ҊO��P\{Pl��M���Oe8N�*2N�
�Nt��Nq�N��aP�lO��O�@�P���PM��O��EN[1MOq�O	h�O�LEO7C�O�?�Oמ Pp��N I�Nk��Oc*�O�=�O�a�O�k�N*�O*FO���O��gOM�O.�OMpiN�	O9��N�]]N76�<�/<��
�o�ě���`B�t��t��D���D���u�u�u�u��t���t����
���
���
��1��1��1��9X��9X��9X��j�ě��ě��ě����ͼ�/��/��/��/��`B�����+�+�+�+�\)�t��t��t���P���,1�,1�0 Ž0 Ž49X�D���L�ͽL�ͽY��y�#�y�#��%��o��o��+��C������������������!)-0+)�����������������������)./)������)+5753)) �Zanqz����zsndaZZZZZZ!)6Ohtvpq}{zt[4(&������������������������������������������
#%/9<GB</#��� #/3<<BFD<4/,##/<@<<5/#
T[hsu~������vtlhe\[T?BMOX[a[YOLB>:??????*6CO\u����zh\6�����������������������	

�������������
#./,/<<2#
 ��]amz�������zmfa[WUV]��������������������").05>BNQNFB>5,)!����$)&!��������/HTa��������zaU?2+*/�#U{�����{c<#
��#/26/#�������������������������

��������5<HQUYZXUHG<93555555��������������������	
	�������������

�������������� )8:6)�����(6@DO[hjxytxth[OB6.(tz�������������vtpqt�()0BenhB)��������������"$������^ajmz�����������zc^^�������������������������)-2)������;BN[bgtuzytoga[SNJB;��������������������[gt���������tog\[WW[��������������������s����������������pnsg������������jfifb\g��������������������`aimommlja_ZXZ``````������������������������������������������������������������sw��������������wptsTUaanornaXUSTTTTTTTTLOT[hlqutptthd[WVQNLz����������������|tzflt������������ytlhf�������������������������



�����������������������������ABNPUV[\[NMIBBAAAAAAbgt|���������{gg`]^b
"#..#
/00<BILMKI=<<0//////���������� ������������������ֺʺɺȺɺϺֺ��������������ùöìàÖÓÑÍËÓàìôù������ÿÿù¡¦²¶½¿������������¾²¦�$���#�$�/�0�<�=�@�I�V�V�O�I�H�=�9�0�$���������������������������������������ſm�`�;�*�����.�G�T�y���Ŀ̿ӿĿ����m�g�b�Z�W�Z�a�g�s���������������������s�g����������������������������������������²¯¬«¤²������������������������¿²�U�H�<�9�8�<�H�I�U�a�n�u�zÅ�z�u�n�a�W�UìáàÙÖÜàììùþ��������ùìììì�/�+�)�#�#�/�<�H�M�U�a�e�a�_�U�J�H�<�:�/�U�Q�U�U�a�f�n�y�zÅ�z�s�n�a�U�U�U�U�U�U�	��ݾҾʾǾ̾׾��	��"�.�3�=�;�.��	����������*�:�C�O�W�O�G�C�6�*���ŭŭŨŭŭŹ������������Źůŭŭŭŭŭŭ���������������	���/�;�D�E�@�;�"�	��������������	��"�"�/�4�9�>�;�9�/�"��	���"� ��"�,�/�9�H�T�a�h�a�_�Z�T�H�E�;�/�"�������������������������������������������t�]�Y�b�k�s���������������������������b�L�_�f�������Ϲ��,�0�"����ù����x�b�����x�i�L�A�D�Y�������������������������6�4�0�5�6�B�H�D�B�7�6�6�6�6�6�6�6�6�6�6�6�4�+�)�&�)�*�6�B�O�S�X�[�b�b�[�O�B�6�6���������������)�-�.�)�%������A�:�;�A�D�M�Z�f�o�i�f�^�Z�M�A�A�A�A�A�AŔŎōŔŖŠŠŭŶŮŭŪŠŝŔŔŔŔŔŔ�������׾־׾����	�����	�������������������������ĿɿͿ̿ȿĿ������������r�f�Z�P�O�R�M�S�r��������ʼռϼ������r��ܹϹù��������ùϹܹ��������������ĳįĽ�����������
��#�<�0�$�������Ŀĳ������!�`�������ҽ���%�6�"��нy�G������r�k�m�s��������!�:�B�B�:�!���ּ���ƸƧƢƘƕƚƧƳ�����������������������	������$�(�$�"���������m�`�W�T�N�I�P�`�t�y�����������������y�m�����������������������ĿȿĿ���������������ٿݿ�����5�A�Z�g�o�l�b�A�5������ھھ߾�������	������	�������ξʾƾľ̾׾����	������	����㻅�q�l�a�`�T�R�S�_�l���������������������~�z²�������
�)�:�Q�U�<�/�����¿FFFFF$F1F2F1F)F$FFFFFFFFFF�Z�W�Z�^�g�r�s�����������s�g�Z�Z�Z�Z�Z�Z�����������x�l�b���������ûл���лû������������ɺ����!�-�8�-�,�����ɺ��r�e�Y�L�@�9�@�L�W�Y�e�r�~�����������~�rŹŭŕŇŀńŔŚŠŭ������������������Ź���������¿Ŀѿտ׿ѿɿĿ�������������������������������������������������������	������� �	��"�/�;�B�H�L�K�F�;�'�"��	�������������������$�5�G�I�B�9�)�����������������������
����
����������D�D�D�D�D�D�D�D�EE	EEEEEED�D�D�D��V�I�>�=�5�:�6�5�=�I�K�[�o�{ǆ��{�o�b�V�h�e�d�h�tāČčĎďčā�t�r�h�h�h�h�h�hĚĒėĚĜĤĦĳĿ��������������ĿĳĦĚE*E#E)E*E7E7ECEPEVE\EUEPECE7E*E*E*E*E*E*����������� �(�(�(������� ) # h d X � H 0 . V - I a G I i I P # ] O ( d Q q 4 R < t � Y & > F U _ 7 U D X I = ( L R O j v Z ) 7 L z B 4 O 5 I p . N ?  p  �  �  �  "  �  �  {  �  �  2  �  4  �  �    �  c  �  �  �  F  R  E  F  D  �  �  �  �    v    7  �  C  �  �    K  9  �      �  E  �           W  �  y    �     �  �  �  �  _<�1;D�����ͼ�1�u�e`B�#�
�ě����ͽ��C����ͼ�1��󶽝�-��h��j�����\)���ͽ8Q��
=���T�����,1��h�t����o�o�u�aG��u�ȴ9���
�T����w�T���8Q콲-�Y���C���\)��1�P�`�H�9�}󶽥�T�}�u�T���y�#������㽛��\���7L���㽲-��hsB�gB�8B�B"UB,�B��B� B�!B ��B|�B�B�ABN�B��B0�VBB~BV�B=�A��BԺBE�B�BV�B&��B��B��B�B-IBzWBe�B�B�/B�B �0BkB-Q�B CBz�BL�B�WB��B	BG�B =B
� B�A���BpaB
�B u�B
ɢB<B��B ��B
�B��B��B:�B>CB	�B0B&T>B��B��B�B�UB@eBRHB��Bg�B ��BBRB<HB��B�ZB]�B0F�BC(B{wB>�A��BŔB4�B7�B�B&ÕBŅB�0B@�B6
B��BWuB��B��B>HB:�B>!B-��B 2�B��B:oB�sB��B	�%BA�B ��B
��B>sA�g�B=�B�zB �bB
�B@�B;�B �B
��B��B�oB>cBF�B	��B?�B&�IA��@De=A�3�A�SBB
��A�ORAl�uA��@��A��A��A̘�A��A��FA[FA�/�A�bA���A��A�A��A��>-A�NA�ԪAإ�AԄA=��A�|VAX�HAv��@�e�>���A�fA6$A ��BO6B	 �Al<�At>�A���AX�kAV�@�%�A�СC��OA�i�@�<=@C�$?�A�)hAy$LA��A�34A�G|A�C�F�B�A��A��0C��-A3�A��@C�eÀ�A�mB
��Aψ�Am9*A��@��A��dAŁA�d�Aœ<AƁJA[ aA�A���A�UA�v!A�6�A��xA�q�>2A���Aב�A؋NA�:�A=-�A�AZ0Aw_Y@�	�>��A慧A�A@TBy�B	CHAkN9At��A��AX�AV��@��aA���C���A���@�<@=��?���A��Ay�%A��(A�vvA��A�XC�?�B|�A��~A��TC���A3�               	                               <   
                     T   <                        "      !   E   2               5      !   #   0            '               -         #                                    1                        )                     )   7   9                        )      #   U   7               %         %   5            %                                                            )                                             !   3   3                        )      #   U   7                        #   /            #                                       Nh��N��mN��O `�N�N��P'+0NC:�NjݟN���N�5�N���N��NB�!O��iO
R�N�S�O��$OP�
O1ކNh=�Oļ�P8�PP[�$M���NѶgN�*2N�
�Nt��Nq�N��aP�lOa3O�@�P���PM��O��EN[1MO_]AO	h�O4P�N�1OJwO�"P8��N I�Nk��N���O�z�O�a�O�k�N*�N��O${O��gN���O.�O@LN�	O9��N�]]N76�  �  )  B  �  	  �  �  �    @  l    �  �  �  �      �  r  �  �  	o  p  |  /  �  $  �  �  l  �  /  �  ]  �  �  �  l     �  1  �  8  B  `  �  r  �  t  �  �  �  �  �  �     �  �  {  �  �<�/<�C���`B�o��`B�t��e`B��t��e`B���
���
��o�u���
�C����
���
���
��1��9X��9X���ͼ����ͼ�j��h�ě��ě����ͼ�/��/��/����`B�����+�+�C��+�e`B�#�
�'#�
�<j���,1�H�9�@��0 Ž49X�D���P�`��%�Y�����y�#��7L��o��o��+��C�����������������)*-))�������������������������������)+5753)) �Zanqz����zsndaZZZZZZ%6B[hljrxtph[B.,+ #%������������������������������������������#)/4<=<7/#
��#/6<<><5/$##/<<<:4/#T[hsu~������vtlhe\[TBBEOW[_[QONB?;BBBBBB*6CO`nssoe\C6*�����������������������	

�������������
#./,/<<2#
 ��]amz�������zmfa[WUV]��������������������#)135:BMDB;5/)######���� %$"��������/<Haj�������zaUH<3./�#Un�����{b<0��#/26/#�������������������������

��������5<HQUYZXUHG<93555555��������������������	
	�������������

�������������� )8:6)�����-;AFO[huvqtvth[OB61-tz�������������vtpqt�()0BenhB)��������������"$������^ajmz�����������zc^^�������������������������#)+.)�����;BN[bgtuzytoga[SNJB;��������������������Z[git|�������tga[ZZZ��������������������qv����������������tqcmt������������yrnfc��������������������`aimommlja_ZXZ``````������������������������������������������������������������sw��������������wptsTUaanornaXUSTTTTTTTTMOW[^hiknpthg[XWRONM��������������������flt������������ytlhf�������������������������



�����������������������������ABNPUV[\[NMIBBAAAAAAbgt|���������{gg`]^b
"#..#
/00<BILMKI=<<0//////���������� ���������������ֺϺ˺Ժֺ���������ֺֺֺֺֺֺֺ�ìèàÙ×ÖÔàìîùü����ùùìììì¦¤¦²¾¿¿������¿º²¦¦�$���#�$�/�0�<�=�@�I�V�V�O�I�H�=�9�0�$���������������������������������������ſ;�$�"�"�-�;�G�m�y���Ŀʿ̿ǿ������m�`�;�g�c�c�g�s���������s�g�g�g�g�g�g�g�g�g�g����������������������������������������¿´²±²¿��������������������������¿�H�>�@�H�U�V�a�f�n�r�n�m�a�U�H�H�H�H�H�HìãàÚØßàáìùý����ùìììììì�/�+�)�#�#�/�<�H�M�U�a�e�a�_�U�J�H�<�:�/�U�T�U�W�a�i�n�u�z�|�z�q�n�a�U�U�U�U�U�U����ݾ׾ؾ����	��"�(�1�3�3�.�"��	������������*�:�C�O�W�O�G�C�6�*���ŭŭŨŭŭŹ������������Źůŭŭŭŭŭŭ���������������	���/�;�D�E�@�;�"�	��������������	��"�"�/�4�9�>�;�9�/�"��	���"�!�#�.�/�;�H�T�a�f�a�]�Y�T�T�H�C�;�/�"�������������������������������������������p�b�]�e�s�����������������������������n�h�_�m�x�����ù�������Թ������n�����{�k�N�F�I�Z�g�����������������������6�4�0�5�6�B�H�D�B�7�6�6�6�6�6�6�6�6�6�6�B�<�6�/�+�0�6�B�O�P�R�[�_�[�X�O�B�B�B�B���������������)�-�.�)�%������A�:�;�A�D�M�Z�f�o�i�f�^�Z�M�A�A�A�A�A�AŔŎōŔŖŠŠŭŶŮŭŪŠŝŔŔŔŔŔŔ�������׾־׾����	�����	�������������������������ĿɿͿ̿ȿĿ������������r�f�Z�P�O�R�M�S�r��������ʼռϼ������r��ܹϹù������ùϹܹ����������������ĳįĽ�����������
��#�<�0�$�������Ŀĳ������!�`�������ҽ���%�6�"��нy�G������r�k�m�s��������!�:�B�B�:�!���ּ���ƸƧƢƘƕƚƧƳ�����������������������	������$�(�$�"���������m�`�X�T�P�J�R�`�m�v�y���������������y�m�����������������������ĿȿĿ�����������������(�5�6�A�J�N�Q�N�L�A�;�5�(�������߾�������	�
����	����������վʾǾʾҾ׾����	�����	���𻑻��t�l�f�c�X�Y�_�l��������������������¿¦²�����
��/�G�N�<�/�
��¿FFFFF$F1F2F1F)F$FFFFFFFFFF�Z�W�Z�^�g�r�s�����������s�g�Z�Z�Z�Z�Z�Z�����������ûлԻܻ�ܻлû��������������������������ɺ����#�$������ֺɺ��r�e�Y�L�@�9�@�L�W�Y�e�r�~�����������~�rŹŭŕŇŀńŔŚŠŭ������������������Ź���������¿Ŀѿտ׿ѿɿĿ����������������������������������������������������������	����	�
��"�/�5�;�A�A�;�8�/�"��������������������$�5�G�I�B�9�)��������������������������
��
���������D�D�D�D�D�D�D�D�EE	EEEEEED�D�D�D��I�C�A�B�I�V�b�o�s�{ǂ�|�{�o�b�V�I�I�I�I�h�e�d�h�tāČčĎďčā�t�r�h�h�h�h�h�hĚĒėĚĜĤĦĳĿ��������������ĿĳĦĚE*E#E)E*E7E7ECEPEVE\EUEPECE7E*E*E*E*E*E*����������� �(�(�(������� )  S 7 X � J , ) l , E a C ? i I P # \ D % \ N q 8 R < t � Y & ? F U _ 7 U D X  2 . D O O j M O ) 7 L u 2 4 8 5 , p . N ?  p  �  /  3  "  �    U  {  S  �  �  4  j  �    �  c  �  �  �  �  �    F  �  �  �  �  �    v  �  7  �  C  �  �  �  K  w     �  �  �  E  �  �  �      W  U  i             �  �  �  _  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  E    �  �  �  
  %  (  )  )  '  %        �  �  �  �  I  �  �  5  �  5    "  ,  3  @  $  �  �  �  �  s  ]  ;    �  G  �  g  �  �  X  X  �  �  �  �  �  �  |  e  H  &  �  �  �  [     �  �  i  	    �  �  �  �  �  �  �  �  t  W  9    �  �  �  d  #   �  �  �  }  y  t  k  a  X  O  G  ?  7  /  '        �  �  �  �  �  �  �  �  �  �  �  �  d  =    �  �  |  `  L    �  �  n  n  k  d  ^  Z  \  f  t  �  �  �  �    j  Q  7      &                �  �  �  �  �  l  J  1    	  �  �  �  �  �  �     =  3    �  �  �  �  �  t  ^  ?  (    �  i      ;  V  c  i  l  k  c  W  J  8  %    �  �  �  �  �  �  �                    �  �  �  �  �  �  o  J  #  �  �  �  �  �  �  �  �  �  �  |  n  `  Q  B  ?  Q  c  k  F     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  e  Y  O  F  >  �  �  �  �  �  �  �  �  �  �  �  a  4  �  �  ]  �  c  m   �  �  �  �  �  w  e  S  @  ,    �  �  �  f  +  �  �  m  )   �             �  �  �  �  �  �  �  �  �  �  �  �  x  j  [      �  �  �  �  �  �  �  �  �  �  �  }  _  ?     �   �   �  �  �  �  �  �  �  �  �  m  O  ;  2  (        �  �  �  D  k  p  g  T  >  &    �  �  �  �  �  �  P    �  �  ]  �  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  @  �  �    	  	g  	m  	Z  	@  	  �  �  �  �  X  "  �  �    u  �  �  @  (  V  p  g  L  (     �  �  q  3  �  �  [    �  n  	  �     �  |  v  q  k  e  `  Z  T  N  H  B  =  7  0  &      
    �       %  )  -  /  .  )      �  �  �  G    �  |  D    �  �  �  �  |  q  g  ]  S  F  4  "    �  �  �  �  �  �  q  U  $        �  �  �  �  �  �  w  b  F  %  �  �  �  P  �  �  �  �  �  �  �  �  �  n  [  N  A  4    �  �  �  �  u  M  %  �  �  �  �  �  x  a  J  4       �  �  �  �  �  m  L  ,    l  ^  P  B  3  #      �  �  �  �  �  �  �  �  k  B     �  �  �  �  �  g  L  -    �  �  �  �  F  �  �  �  �  [  8  �    '  /  +    �  �  �  �  �  �  j  B      ;    �  V  �  �  �  �  }  q  P  $  �  �  x  J  !    �  �  �  x  ?  �  �  ]  T  D  '  �  �  �  C    �  V  �  �  q  6  �  �  ?  �  �  �  �  �  ^  1  �  �  �  N    �  �  i  ;  �  }  $  �  �  D  �  z  l  a  U  @  %    �  �  �  a  -  �  �  }  /  �  �  ,  �  �  �  �  �  �  �  s  ]  G  0      �  �  �  �  c  B     i  l  g  `  V  J  ;  '  
  �  �  �  �  \  +  �  �  w    �       �  �  �  �  �  �  p  P  /    �  �  �  n  G  "    �  �  ,  >  F  Y  �  �  �  �  �  �  �  U    �  ?  �    �  e    %  .  0  1  0  *    	  �  �  �  �  v  H    �  ^  �   �  �  �  �  �  �  �  �  �  �  l  ?    �  �  $  �  Y  �  x  �    0  7  7  -    �  �  �  �  a  7  	  �  |  $  �  j  �  �  �    2  ?  @  3    �  �  b     �  �  k    �  ]    �  {  `  C  %    �  �  �  m  C    �  �  �  J    �  R  �  �    �  �  �  t  Y  >  !    �  �  �  |  U  -    �  �  r  9   �  �  �  �  �      D  q  l  Y  :    �  �  �  Y    �  U  �  �  �  �  �  �  �  �  �  �  z  `  =    �  �  ;  �  �  �   �  t  X  <       �  �  �  }  C    �  �  �  �  �  m  G    >  �  �  �  x  l  `  Q  @  +    �  �  �  �  b  <    �  �  �  �  �  �  �  �  �  v  l  b  X  P  K  F  A  <  5  -  &      p  �  �  �  p  S  1    �  �  �  q  ]  h  m  F  "    �    &  `  �  �  �  �  �  �  �  �  �  L    �  X  �  f  �  �  �  �  �  u  a  K  4       �  �    7  �  �  K    �  �  j  +  �  �    z    �  �  �  w  \  =    �  �  �  o  <  �  �  !     
�  
�  
�  
�  
[  
%  	�  	�  	j  	  �  s    �  6  �  �  �  �    7    #  �  p  W  9    �  �  �  T    �  �  6  �  T  �  �  �  �  �  �  �  �  �  v  X  ;    �  �  �  �  �  z  `  F  {  g  R  <  %      �  �  �  �  �  �  �  }  R     �  �  8  �  �  �  �  �  �  �  ~  a  =    �    %  �  n    �  �  S  �  �  �  �  �  �  �  �  �  �  �  �    q  Z  C  +     �   �