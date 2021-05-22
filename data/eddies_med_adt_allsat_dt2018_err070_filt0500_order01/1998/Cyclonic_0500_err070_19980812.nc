CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�\(��        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�lf        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\   max       <49X        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�z�G�     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min                  max       @v�p��
>     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q`           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @���            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �1'   max       ;ě�        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B4�h        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4��        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�6�   max       C���        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          U        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          I        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�lf        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���v   max       ?���#��x        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\   max       <49X        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�z�G�     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�
=p�     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O            �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�]�            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E[   max         E[        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?���#��x     �  ^�                                                   :         #      $   &   U         -          M                           $   L            !                     7                         
   C   ,   !   	         ,      3            NW��M���NbC�N0�jN��3O�ZlNXI�N.��O��O'$.O��jN$COe�N�/�O�b�N���P�lfN��[OQ�P�pOA�O�gO��P���O'XN�IQP�Y�N,,�P[PS�NhzCO���O��9O+eOܫO<�'NO\�O%��O��1PI�O��]O�*O�bOh�JO���O�NȈiN`:1N��O�0;P'	HNS@;N��ZO��hN���O��VO.'rO�N�-O�AO���O6$�N� �On�N��2O�6�O��\O��vN-�_N�'�O
 *N�Ȼ<49X;�`B;D��%   ���
�o�o�o�#�
�T���T���e`B�e`B�e`B�u��o���㼛�㼛�㼛�㼛�㼣�
���
��1��1��j�ě����ͼ�����/��h��h��h�������o�C��\)�\)�t��t���P���#�
�#�
�'49X�<j�L�ͽL�ͽL�ͽP�`�]/�]/�e`B�ixսixսq���}󶽁%�����7L��t���t����P���w���罰 Ž�-��-�\fhilstw����~wtphffff�����������������������

�������������������������������u������������~xwuuuu	'/7;?>@>?:7;/"		KOU[_ahf][ZRONKKKKKKotvv������{toooooooo)/HUajaajlkkU<.)($$)��������������������*6CO[c`[VNC6*Z[hntth[WZZZZZZZZZZ����������������������

�������H\cnx�������naUHC?AH�����������HTmp���������noaPKEH������������������������������������������
6BO[hru}sh[B)�5<BO[efeaa`^[YWOEB45�#/8<BDH</#
�����������������������������
 ����������������������������������������������������
#Qd{��������{<#�����������������������:?HTz����|vhaTH@<:9:��
0UejokbI<#��������������������������������

�����������
���������������� ���)4:BN[g�����t[N5))$)5BN[a[[SNEB5)"��������������������������������������������������������������**���������.9B[gu���ztWNIB=85..V[cgt��������tgdb[UV<BCJOV[hjqojihd[OGB<��
"/<HQURK<#
����������������������uz}������������ztpqu��������������������ABO[[\\\[ODJCBAAAAAA`adnrwna]\``````````x~��������������zvux��������������������")6<<6/) 	)46>>6)'						LN[gt�������tg^[WNKL�����������������������)6:91*����������������������������������������������EHKMUakha^XUMHCHKHEE������������������	)19?A54,)	��������������������{������ztnldiny{{{{#'/<HKJMUageaU:/)!#lnpxw{�{unljgehhjlAGN[t����ytpj[NB?>>AZht�����������wsg][ZTWanz���������znaZTT;<EHNMHF<:88;;;;;;;;�����������������������#&&
������	
 
						�H�A�<�/�#�"�#�/�/�<�?�H�U�X�U�K�H�H�H�H�4�1�*�4�A�M�R�M�I�A�4�4�4�4�4�4�4�4�4�4��s�s�q�s���������������������ǔǋǔǔǡǭǸǯǭǡǔǔǔǔǔǔǔǔǔǔ�Z�V�Z�f�g�s�u�������������s�f�Z�Z�Z�Z�	��������������������"�;�H�K�A�;�2�"�	�m�d�`�^�`�m�y�����������}�y�m�m�m�m�m�m�G�<�;�.�'�+�.�;�A�B�G�M�G�G�G�G�G�G�G�G��������������������������������������������������������$�*�0�2�4�0�$������.��	���پپ�����"�.�6�;�?�@�>�;�.�����������ĿƿпͿĿ����������������������������������������ʾ̾̾ʾɾ¾��������������������������þʾ׾�޾׾ʾ���������	�������	��"�;�H�U�X�Y�[�X�N�H�;�/��������s�p�s�~���������������������������{�tĜĦĽ�������<�U�{�~�l�U�#���ĳč�{ŇņŇŔŖŠŭŹžŹŶŭŠŔŇŇŇŇŇŇ��������������������������������������m�H�7�1�%�!�	��"�;�H�a�b�\�m��������������������������������	���������������T�H�2�)��#�)�<�H�U�f�xÀÇËÇ�z�n�a�T�����x�l�`�U�\�Z�_�l�x����������������������������������	�H�a�~�����y�y�H����˻���������!�-�4�:�:�=�D�:�-�!�������������!�,�)�!�����������������������z�Z�A�������>�s�����������������Y�U�M�Y�f�m�r�|�r�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�������߿�����5�Z�y�����s�A�5���н����i�c�s�����н����������������ݽп.�-�.�;�B�G�T�`�j�m�o�m�`�T�G�;�.�.�.�.��������ŹŵųųŹ����������������߻ܻӻһֻֻϻлܻ����$�'�������ܿ.�"��	�������ݾ�����	���#�/�.�.��׾ʾ������˾����	�������	�����I�H�=�:�7�7�6�5�=�V�W�b�h�o�p�n�i�b�V�I�ʾ¾��������ʾ׾ݾ۾׾ξʾʾʾʾʾʾʾ��)� �����)�*�5�B�N�[�b�c�g�[�N�B�5�)�ݿѿ��������������Ŀѿݿ�����������ݼ]�^�u����ʼ���!�&�"������ּ��r�]�z�m�g�e�h�m�z�������������������������z���������������������������������ѹܹҹϹù����������ùϹܹ������������������#�%�)�6�B�O�[�I�B�6�0�)���Z�N�A�(�����*�N�Z�f�s�������s�i�^�Z�лƻû����������ûлܻ����� ������ܻ�Ç��z�x�x�zÃÆÇÓÕØÙÞàâàÓÇÇ�û»��ûûлܻ����ܻл̻ûûûûû��5�4�5�<�A�N�X�X�N�A�5�5�5�5�5�5�5�5�5�5ƧƚƁ�m�h�\�R�\�sƁƚƬƳ������������Ƨ�u�p�x���������ɺ������� ��ֺ����u����������������������������������������ìèäåìóù��������ýùðìììììì�6�1�*�1�6�=�B�D�\�uƁƌƘƒƁ�y�h�T�C�6�U�P�R�L�U�Y�b�n�t�y�n�b�U�U�U�U�U�U�U�U�'��������'�4�@�M�R�]�c�c�^�M�4�'à×ÓÐÓÔÐÓàìíù����������ùìà�������������������������ɺ˺ҺպɺǺ���E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������ĿīĪĵ���������
���$�(�#������Ě�t�h�`�[�X�]�h�tāčĢĦĳĵĺĺĳĦĚ��������������$�0�4�7�6�3�0�$�������������ĿȿοϿѿҿѿĿ�������������E�E�E�FFFFF$FBFVFcFoFrFhFJF@F1FE�Eễ�~�����������ûл׻ܻ�ܻлû�����������������¼¶¿�����
�#�/�<�T�a�`�H�#����ĚčĉĈčĚĦĳ������������������ĳĦĚ�����������������������׾����ܾ׾ʾ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��!��� �!�.�:�G�Q�L�G�<�:�.�!�!�!�!�!�!�l�b�l�y�������������ƽ��������������y�lEEEEE*E7ECEPEREPECEBE7E*EEEEEE J L S Y \ p S e ) 6 N Z ! D % b n X 2 q � E 3 x - : o G t 2 | 4 4 b 9 7 < ? Q e . A 1 � G S g K K m 1 < r L K f : @ p . 1 D L � y o S + F K z a  s  !  {  T  �  :  �  s    o  w  V  O    �     �  �  I  �  9  {  `  �  0  �  �  I  6  �  �  m  S  `    �  m  m  ^  �  v  d  +  D  �  u  "  �  G  �  �  s  �  L  �    }  "  �    $  �  �  �  5  �  R  a  \  �  �  �;ě�;�o�o�ě��#�
����D���49X�o���\)��C��ě������h��1���-������P�Y���w�e`B�ixս�
=�\)����O߼��m�h���o�m�h�H�9���q���}�t��@���hs��G��q���'m�h��hs�������D���L�ͽH�9��+��
=�}�}󶽗�P�q����-�������-��C��o����ȴ9������Q콟�w���;d�1'�Ƨ�����vɽ�l�B?=B��B�FB
�DB�UA��B5�B��B#�B�kB0/�B��B!��B$!B��B�_B eB��B_\BPB�B��B H�B��B+��B"��B&�B g�A�>=B%�B8�B;B#!B�B�tB�B4�hB�\B�B-M B��B	��B<�B�BxMB��B�QB�]B|�B �(BwKBC�BtB	e)B;B��B�B �/B��B�BF2B\wBg
B��B(YdB	H�B
�Bs�B�BTBnB'1B>�B��B|�B
��B:�A���B?DB��B?�B��B0AB��B"AB$?oB�0B?3B cB��BHSB�B��B��B ?B��B+�PB"�JB&��B �aA���B%�SBA�B?iB#=�B?�B	+8BI�B4��B�RBAB-?�B�zB	�|B?kB@B?�B��Be5B��B��B �JB@~BA�B�mB	<�B
��B�B�B ��B�gB>8B?)B~jB�lB?�B(7�B��B	��BH�B�.BxB��BAjAË�A:��AF�#B�4ABلA���Al�\AcM�A���B�A[��Ax�mAL1�ANOtA��dA���A�FsA�s�A��YA��A���Aň�@���A��[@f�[@a8A�k+@��A���A"a�Ae��A���@���AZ��AXIRBq*AQ{A�8�Ax��@���A���A���>�6�A�ػA�zs@�ϬA�:�@���A�٩B�@.1A�TA�UB�)A�Z�@�uyA�B@#�C�hA�TkA�C}B	e!Ax��C���@�2+A���A�[AN��C��,A��AE%C���A�d;A:��AF�B�GADlLA��cAm�Aa'�A�čB	]A[��Ax��AL�AL�pA�{�A���A�u�A�A�{A��A�� A��@��A�\�@d�9@_�!A��>@���A���A!7�Ae�A��X@���AY�AY�B�5APF\A�K�Ay A��A���A��>��A׀A�s�@�z|Aɡ@��YA�SB��@3"eA�=kÀB�yA���@�_�A�ȳ@+r�C��A�|�Aݖ�B	HmAx+C���@�*zA�r�A��AN��C�ˮA�gA ��C���                                                   ;         #      %   &   U         .      !   M            	                %   L            "                     8                         
   C   ,   "   	         ,       3                              %                                 I         -      #      A         E      +   /               #               7               !                  -               #            #            %      )   !                                 %                                 I                     1         ?      )                                 3                                 +               #                              '                  NW��M���NbC�N0�jN��3O�ZlNXI�N.��O+]AO'$.O��N$CN��N�/�O�b�N���P�lfN��[N�#lO<S�N���Nܿ$Ol�PX��NC-zN�IQPs��N,,�O��LO���NhzCO��ZO�r�O+eO�.O<�'NO\�O%��O{g�P7PgO�Q�O�*N�/%O0�)O�
�O�sNȈiN`:1N��O�0;P"e�NS@;ND��O��hN���O��VN�@.O�Np��O{��O���O�N� �O�N��2O�t�O{�O��vN-�_N�'�O
 *N�Ȼ  R  �  �  �  -  �  +  Q  �  �  �  �  }  �  P  ;  x  �  h  o  V  ;  /  	Y    n  !  �  {  �  �  �  X  �  W    �  �  z  �  �  /  �  /  �  �  �  �  �  �      6  /  �  �  r  �  >  	-  	�  	8  �  �    �  >  �  y  �  �  �<49X;�`B;D��%   ���
�o�o�o�u�T���e`B�e`B��C��e`B�u��o���㼛������+���ͽ�P�����\)��`B��j�������ͼ�/�y�#��h�+�����+���o�C��t��#�
��w�t���w�,1�,1�''49X�<j�L�ͽP�`�L�ͽY��]/�]/�e`B�y�#�ixսu���
��%��t���7L���㽓t������������罰 Ž�-��-�\fhilstw����~wtphffff�����������������������

�������������������������������u������������~xwuuuu	'/7;?>@>?:7;/"		KOU[_ahf][ZRONKKKKKKotvv������{toooooooo*/<=HNU[`_WUHD<//-)*��������������������*6COX\\YTLC6*Z[hntth[WZZZZZZZZZZ����������������������

�������H\cnx�������naUHC?AH�����������HTmp���������noaPKEH����������������������������������������	)6BHHDB6)?BEO[`ba`^][SOKDB>??#//782/)#������������������������ ���������������������������������������������������� 
#Sgn������{<#�� ��������������������;@HT����{uhaTHA?=;:;#0IRUZ\\UI</#������������������������

��������������
���������������� ���')29>BN[gt~����t[5)'$)5BN[a[[SNEB5)"��������������������������������������������������������������'%����������0<BN[gnt}�~t[NB@;820V[cgt��������tgdb[UVKOY[hionihg[OJCDKKKK
#/HKRPI</#
 ��������������������uz~�����������zupruu��������������������ABO[[\\\[ODJCBAAAAAA`adnrwna]\``````````x~��������������zvux��������������������")6<<6/) )156:6)(LN[gt�������tg^[WNKL�����������������������)6:91*����������������������������������������������FHLOUajfa\WUNHFFFFFF��������������������	)19?A54,)	��������������������{������ztnldiny{{{{#*/9<EHMOMHB<//+(#lnpxw{�{unljgehhjlAGN[t����ytoj[NB@>?A[it�����������ytg^[[TWanz���������znaZTT;<EHNMHF<:88;;;;;;;;�����������������������#&&
������	
 
						�H�A�<�/�#�"�#�/�/�<�?�H�U�X�U�K�H�H�H�H�4�1�*�4�A�M�R�M�I�A�4�4�4�4�4�4�4�4�4�4��s�s�q�s���������������������ǔǋǔǔǡǭǸǯǭǡǔǔǔǔǔǔǔǔǔǔ�Z�V�Z�f�g�s�u�������������s�f�Z�Z�Z�Z�	��������������������"�;�H�K�A�;�2�"�	�m�d�`�^�`�m�y�����������}�y�m�m�m�m�m�m�G�<�;�.�'�+�.�;�A�B�G�M�G�G�G�G�G�G�G�G��������������������������������������������������������$�*�0�2�4�0�$������.��	����۾ܾ�����"�.�5�:�>�>�<�;�.�����������ĿƿпͿĿ������������������������������������������žž��������������������������������þʾ׾�޾׾ʾ���������	�������	��"�;�H�U�X�Y�[�X�N�H�;�/��������s�p�s�~���������������������������{�tĜĦĽ�������<�U�{�~�l�U�#���ĳč�{ŇņŇŔŖŠŭŹžŹŶŭŠŔŇŇŇŇŇŇ������������������� ������������������l�a�T�H�A�A�B�A�H�T�a�b�q��������z�q�l�����������������������������������������H�A�<�7�<�<�H�U�a�f�n�r�p�n�a�U�H�H�H�H�x�w�m�b�b�b�l�������������������������x�������������������	�"�Y�`�X�H�;��������������!�&�,�+�!�������������������!�,�)�!�������������������������{�Z�A�3���N�s�������������������Y�U�M�Y�f�m�r�|�r�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y������������5�Z�w�����s�Z�A�5�������z�u�������������ȽѽҽнҽνĽ������.�-�.�;�B�G�T�`�j�m�o�m�`�T�G�;�.�.�.�.��ŹŷŵŶŹ����������������������ƻܻջӻԻ׻׻ѻܻ�����#�&�������ܿ.�"��	�������ݾ�����	���#�/�.�.������׾ʾƾ¾ľоھ����	�����	���I�H�=�:�7�7�6�5�=�V�W�b�h�o�p�n�i�b�V�I�ʾ¾��������ʾ׾ݾ۾׾ξʾʾʾʾʾʾʾ��)� �����)�*�5�B�N�[�b�c�g�[�N�B�5�)���ݿѿ����������������Ŀѿݿ����������e�h�x�����ֽ��#�$� ������ּ���r�e�z�m�i�h�h�k�m�q�z���������������������z���������������������������������ѹù����������ùϹܹ������ܹϹùùù�������!�(�)�6�B�O�Y�O�G�B�6�-�)���(�����(�5�A�Z�s�~����p�g�a�Z�A�5�(�лǻû��������ûлܻ����� ������ܻл�Ç��z�x�x�zÃÆÇÓÕØÙÞàâàÓÇÇ�û»��ûûлܻ����ܻл̻ûûûûû��5�4�5�<�A�N�X�X�N�A�5�5�5�5�5�5�5�5�5�5ƧƚƁ�m�h�\�R�\�sƁƚƬƳ������������Ƨ�v�q�x���������ɺ����������ֺ����v����������������������������������������ìææìùü��������üùìììììììì�6�1�*�1�6�=�B�D�\�uƁƌƘƒƁ�y�h�T�C�6�U�P�R�L�U�Y�b�n�t�y�n�b�U�U�U�U�U�U�U�U�'��������'�4�@�M�R�]�c�c�^�M�4�'àÚÓÒÓÖ×àìùü����������ùìàà�������������������������ɺ˺ҺպɺǺ���E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���Ŀĸĳĳ��������������
�����������Ě�t�h�`�[�X�]�h�tāčĢĦĳĵĺĺĳĦĚ��	����������$�0�1�4�3�0�0�$��������������ĿȿοϿѿҿѿĿ�������������E�E�E�FFFFF$F1F=FJFOFJFHF>F=F1F$FE����~�����������ûл׻ܻ�ܻлû�����������������¼·¿�����
�!�/�<�S�`�Z�H�#����ĚčĉĉĈčēĚĦĳ��������������ĳĦĚ�����������������������׾����ܾ׾ʾ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��!��� �!�.�:�G�Q�L�G�<�:�.�!�!�!�!�!�!�l�b�l�y�������������ƽ��������������y�lEEEEE*E7ECEPEREPECEBE7E*EEEEEE J L S Y \ p S e  6 L Z % D % b n X . I R " 1 l 6 : e G v  | * 3 b 9 7 < ? N _ * A 4 u L I g K K m 0 < d L K f 1 @ Z ! 1 < L m y o A + F K z a  s  !  {  T  �  :  �  s  g  o  U  V  �    �     �  �  �  �    �  �  `  b  �  �  I    Z  �    '  `  �  �  m  m  L  �    d  �  �  v  B  "  �  G  �  �  s  �  L  �      "  �  �  $  C  �  m  5  �    a  \  �  �  �  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  E[  R  C  5  &      �  �  �  �  �  �  y  `  H  0     �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  d  M  6       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  d  P  <  )     #  &  !    �  �  �  �  �  -  #        �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  u  j  Z  F  -    �  �  �  �  �  �  ~  h  J  $   �   �  +  -  0  2  5  7  8  :  ;  <  >  A  C  F  H  J  L  N  P  R  Q  W  ^  e  k  r  y  z  y  x  v  u  t  p  g  ]  T  K  B  9  @  y  �  �  �  �  �  �  �  �  �  �  u  R    �  K  �  o   �  �  �  �  ~  o  ^  J  2    �  �  �  �  H    �  p  #  �  }  �  �  �  �  �  �  �  �  u  b  K  .    �  �  �  E  �  �    �  �  �  �  �    	    �  �  �  �  �  �  �  �  �  �  �    ;  Q  g  n  u  z  }  {  v  n  b  S  <  "  �  �  �  p  ,   �  �  �  �  l  Z  U  P  K  D  9  .  $       �   �   �   �   �   �  P  I  :    �  �  �  �  �  �  o  N  +    �  �  �  H   �   �  ;  >  A  E  D  2  !    �  �  �  �  �  �    i  S  >  (    x  U    �    �  �  ~  E    �  �  �  �  i    �     �  N  �  �  �  �  �  �    w  n  g  a  [  U  O  D  3  !    �  �  H  U  ]  e  c  [  ]  e  _  K  4      �  �  �  �  ]  �  f  �  ~  z  ~  �  �  9  k  m  e  _  C    �  �  C  �  ~  �  �  �  �    4  G  S  V  T  P  G  :  %    �  �  �  �  o    �  �  �    y  �  �    #  5  ;  4     �  �  I  �  M  �  �  �       +  /  (            �  �  �  u  H     �  $  ^  �  8  �  	-  	X  	S  	=  	  �  �  �  E  �  �    �  U  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  J    �  n  f  _  V  J  =  /      �  �  �  �  �  s  I    �  �  �       �  �  �  �  p  L  "  �  �  �  n  $  �  �  f  '  �   �  �  �  �  �  �  �  �  �  �  �  �  �  s  c  O  <    �  �  3  {  z  v  l  Y  K  ;    �  �  q  (  �  �  �  �  i  -    �  =  y  �  �  �  �  �  �  �  �  �  �  O    �  "  �    4  %  �  �  �  �  |  o  b  c  j  q  x    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  1  �  �  �  �  U    �  `  �  %  X  X  N  A  3  $      �  �  �  �  `  5    �  �  [  D  �  �  �  �  �  �  �  �  �  r  X  <      �  �  z  /   �   �   P  ;  L  W  T  M  A  /      �  �  �  �  �  i  +  �  i  �   �    �  �  �  �  �  �  z  H    �  �  S    �    m  �  �  "  �  �  �  �  �  �  �  �  �  �    p  a  R  D  7  +        �  �  �  �  |  l  Z  K  :  (    �  �  �  �  �  }  ^  2  �  o  z  p  `  L  3    �  �  �  x  4  �  {  
  �  9  �  {    �  �  �  �  �  �  �  `    �  v  	  �    k  �    \  �  �  �  �  �  �  �  �  �  ~  X  )  �  �  �  d  5    �  �  r  �  /  (  !          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  f  h  l  m  g  [  K  .    �  �  (  �  �  !  (    �  �    �  �  �  W      �  V  �  �  �    �  �  �  �  �  �  �  r  �  h  H  $  �  �  �  �  �  C  �  X  g  �  j  S  9      �  �  �  r  <  �  �  D  �  )  �  �  (  �  �  �  �  �  �  �  �  �  �  �  o  S  8    �  �  �  $  �  �  �  �  �      #  &  $  !        �  �  �  �  �  �  �  �  �  |  s  i  `  V  K  ?  3  '        �  �  �  �  �  �  �  �  �  �  Z  ?  F  &    �  �  �  �  b       �  �  .   �  �      �  �  �  �  �  ]    �  �  %  �  t    �  �  �      �  �  �  �  �    i  Q  =  &    �  �  �  �  i  =  �  V  �    +  3  5  4  I  `  \  X  R  K  A  ;  7  >  g  �  �  �  /  )      �  �  �  �  a  ?    �  �  �  }  E  �  �  R  �  �  �  �  }  s  p  l  h  c  ]  W  Q  I  A  9  1  #      �  �  v  \  ?  '  $  	  �  �  �  �  B  �  �  I  �    �    (  Z  \  ^  h  o  ]  C  &  	  �  �  �  w  H  �  �  f  .  �  �  �  �  �  �  �  �  �  �  m  W  <    �  �  �  o    �    h  +  4  =  )    �  �  �  �  �  k  V  H  <  6  .  $        �  V  �  �  	  	,  	(  	  �  �  �  Q    �  1  �  �  �  �    	�  	�  	�  	\  	'  �  �  Z    �  �  /  �  ^  �  �  C  �  �  V  �  	  	*  	5  	8  	0  	  	  �  �  z  :  �  z  �  0  `    �  �  �  �  �  �  �  t  d  T  C  2      �  �  �  �  �    X  2  �  o  S  �  �  �  �  �  �  �  X  :    �  �  ]    �  6          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  d  E    �  �  T    �  b  
  �  l  9  �  r  �    �    6            �  �  �  �  f  3  �  �  n    �    �  �  �  |  <    �  �  _  +  �  �  z  8  �  �  B  �  B  �  �  y  `  J  9    �  �  �  ~  ]  J  F  J  X  f  p  z  �  �  �  �  �  �  �  �  �  �  v  e  N  6      �  �  �  �  c    �  �  �  �  x  k  _  R  B  0      
    �  �  �  �  �  �  �  �  �  s  X  5    �  �  c  ,  �  �  �  =  �  (  �  !  �  �