CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�/��v�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M矫   max       Pm�$        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       <49X        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @F@          @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v~fffff     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @��             8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �
=q   max       ��`B        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�[�   max       B0o�        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�gk   max       B0A�        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�wy   max       C��)        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�N�   max       C���        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          r        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P(V7        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�o���        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       ;o        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @F5\(�     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v~fffff     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q@           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�t�            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?�o���     �  ^�      	   +      
                  (         r         
         6      %                              l                            
                  #   $      
            &         $                           #   ,               O��9N�+�O�S�N���N��N��N��O�u?N�*M矫O��zOv��N��>Pm�$P��OT�HOT O�"6NNPvO	�O�sON��Oi�rN�$\NM,O�`N�mN[��N#�BO��PhR�N�.�N� �N���O�l�OmGO���N8�O5P�N��N�XO[��O2,O,�N!�LOO�jO�2�N��+N��O��O���O�gO�[�Pm�O��sO�Z�N?��O��Oy;`O%�.O@�JO?MNA�:Nk�HO���O���N��N�PN��}N�{�N1�x<49X;o:�o�D����o�ě���`B�o�t��e`B�e`B�u�u�u��o��o��o��C���t����㼛�㼛�㼛�㼛�㼣�
���
��9X��9X������/��/��`B��h��h���o�o�o��P��w�,1�,1�0 Ž8Q�8Q�8Q�@��@��H�9�H�9�T���T���T���Y��Y��m�h�u�y�#�y�#�y�#��o�����������O߽�\)���-�������T��1�Ƨ��`Bgt������������vqlhfg*59BJN[\b`[NB<5/****��������������������NO[dhtuvutplh[YOOLJN;<AIUbdba[UPJI??<;;;jnsz��������zronjjjj��������������������pt��������������tnip���������������;BOQVPONB?;;;;;;;;;;��������������������ht�����������thb]^ah��������������������#0<U]g{�tfUI<52*)"#[gt����������~o[NOS[��������

��������������������������%-6CO\bbYOIC6*��			������������������� ����������HOUanz������ztnaUPHH
(.-+,+)#�����������������������������

������Z[fgtv���}utg[ZZZZZZW[ghnlg[SSWWWWWWWWWW"/CHT\muyxmaTHC/%����������������������������������������')312+)&46BO[hlijkkph[OD@744����-.'*+#�����U[glqtuwwutqgb[XTSRUqtw�����������tsqqqq36BCJOOSPODB@641//33DNt���������tng[\NDDBHOZhtsxtqvth`]VOB9B-2:GVb{�������{bI<0-P[gtxxtg[QPPPPPPPPPP��#03020#
���������	���������������� ����������������)6<A=6)�����������������������������

�����������������������������������	������������������������������������������������������������������������������y|}���������������{y�����	���������M]t�����������tg_ULMru|���������������wr|���������������}}}|#/<HOSa[YVNH</-#����������������������������������������;ADDHUanuxunaUH<504;}����������������~}w|�������������zvuw
#+<?<963/#
	QUalmbaUTJQQQQQQQQQQ������������������������������������������
#+/6<BC5/
��������������������
 ����558BENPTTTNB;6545555ggkqtx���������wtgggaY^anz{zvnaaaaaaaaaa�����¿ÿǿѿݿ�����������ݿѿĿ��g�\�Z�X�Z�\�e�g�s�����������y�s�g�g�g�g�/������������6�<�C�M�T�U�a�z�z�a�U�H�/�m�k�d�`�^�]�`�m�y�{�����������������y�m�ʼż����������Ǽʼּ߼�����߼ּʼʺֺԺɺĺ������ɺֺ�������ٺֺֺֺֿ��ݿؿٿݿ�����������������������ݽ׽ݽ������#�'�%�!���������������������� � ���������a�]�_�a�n�z�~�z�q�n�a�a�a�a�a�a�a�a�a�a������
��!�-�F�S�_�o�|�~�x�l�_�-�!����׾ʾʾξվ׾����	���"�#��	�����׿.�'�+�.�.�;�G�H�K�I�G�;�.�.�.�.�.�.�.�.���|�z���������A�Z�q�y�v�f�A��ݽ������������������	��"�)��'�/�H�J�;�/�	����������������������������������������y�s�m�h�e�m�y�����������������������	�����ھӾվ�����&�.�8�:�6�*�"��	�	������������	�����	�	�	�	�	�	�	�	�/�)�&�$�&�-�;�H�T�m�������������m�T�H�/��������������������������������� ������ֺϺɺźȺֺ���!�-�3�:�>�:�5�-�!��ֻлŻû������ûĻлܻ�����ܻлллоʾþ������������ʾ׾����������׾ʾ���������(�4�6�:�8�4�(�����Z�Q�R�Z�f�s�v�|�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z����������������������������������������������	���������������������������������������������лλû��ûлܻ���߻ܻллллллл���������������������������������������������f�F�4�)�"�4�M������ʼ˼Ƽ����������u�i�uƁƎƖƚƧƳ������ƺƳƬƧƚƎƁ�u����¿¸¿�����������������������������˻S�P�R�S�_�k�l�x���������������x�l�_�S�S�����������������������0�5�I�8�)�����ֹù������ùܹ������ �������������������s�n�^�[�m���������������������������������	�
���	������������������������������������$�'�-�-������	����������	�� �"�-�.�"��	�	�	�	�	�	ā�{āĄċĈčĚĦħĳĴĳįĦĚčĈāā�T�O�H�;�7�3�0�1�5�;�H�T�`�g�m�r�r�m�a�T���������������������������������������	���*�+�6�8�@�>�6�*�����������������������������������������Ѻ�������������'�3�4�9�<�=�3�'��������úĺɺ�����-�:�S�N�:�!���ֺ��T�R�N�S�T�`�a�m�t�y�t�m�a�Z�T�T�T�T�T�T������������������������������������������ ������"�(�-�5�8�=�8�/�*�(����@�4�'�����8�M�Y�f�r�����|�r�g�Y�M�@�������������ʼ�����	��������ּ�����ĳĦěĖēĕĚĦĳĿ����������������Ŀĳ�0��
�����������������
�0�L�f�u�t�<�0ŭţŠŔŋŕŠŭŹ��������������������ŭE�E�E�E�E�E�E�E�E�FFF$F1F=FFFAF=F(E�E������������������������������������������ĿĳīĳĿ�����������������������N�5�(�����&�5�A�N�\�c�e�j�v�s�g�Z�N�I�?�=�6�6�=�?�I�V�b�o�{�~�{�v�o�d�b�V�I����ƳƧƚƎƆƔƚƧƳ�����������������̿�ݿѿοϿѿؿݿ��������������Ŀ¿��Ŀѿݿ޿ݿݿѿĿĿĿĿĿĿĿĿĿ����������
�����$�$�$�������������6�)������)�6�B�O�[�h�t�w�h�`�T�B�6ED�D�D�D�D�D�D�EEE*E/ECEFEAE7E1E*EE������������������������`�X�S�Q�R�S�_�`�l�y�������������|�l�d�`ùõìæìôù������������������ùùùù�/�/�#���
��
���#�/�<�?�>�<�:�0�/�/��ܻлͻǻɻлӻܻ����������� 4 E J 7 F I 3 2 I Q R  V c j 7 S / { 4 z A . & . @ ] Z # F a L i 7 D V N M e _ 0 Z @ 7 ; b M g M < � \ F 2 a - 0 S ] I ' q & : n 4 1 J ] M ) L  T    `    �  �  �  
  �  /  o  �  �  �  �  �  5    D  �  �  �  �  �  �  S  Q  S  b  E  �    M  �  �  2  
  d  �  �  �  �  �  X  x  -  �  ]  �  �  �  ~  Y  d  �  #  �  z  _    _  �  �  K  �  G  \  V  ?  �  �  _��C���`B�,1�o�T����o�49X�ě����㼣�
�Y���P����o���\)���ͽ49X��1��t������aG��\)�#�
�ě����ͽ#�
�ě��+���H�9�
=q�'C��8Q�T����o�u�'Y��P�`�L�ͽ�7L�e`B�m�h�L�ͽ��Y��m�h��7L��\)���罺^5������P�\������w���w��-���T���hs������
=�����罾vɽ����G����mB
ǁB��B B�B&��Bh�B �B|FB�Br}B,i�B�B �B&��B
�B�B*�B0o�A�[�BB��BQ�B�_B#�tB	p�B�!A�B*cKB,�iB7:B�B�;B	"B�&B�B	��BP�B(kB	EB$c?B1zBsB��B�;B�PB�VBE�B">B��BcB@�B*<�B-aB	�ZBB/JB�4B*�B��B�B�B ��B�8B:BMB2�B%]B�B�@B�.B
B�B�3B
��B'9B?fBk�B&��B[�B �cBa�B�)B�xB,�jB�_B ;B'L!B	�4B@�B*A�B0A�A�gkBs4BX�B~IB�B#�B	��B��A�{CB*��B,G�B2B��B?LB	9�B�@B;bB	B�B?�B)>HB	8�B$@1B(B��B�EB�sB�cB�\BJ4B#;�B|�BmBQ?B)�qB-��B	�B��BABG�B>UB��B2�B3�B @RBEBB?iB�B;�B�pB��B�"B�$B
B4B_�A~ڊA�U/A���Am��A �@=�A A0b�A�hA�M$@$�AWUAbq_A-�%A�-�A�&HApLAZEA�$�A���A�aQ@W�@�I�AToHA6f1AA,A��A�.�A���@��VA�UJ@��SB��A���@��1A�:�>�wyA��eAY��@���A\�A�,�A�5�A���A�g�A���?�j@X;A��SA���A�b7@�٦A �A�!	A�^dA� +C��)BxNA��A��'B��B�iA�-�Az��BחA�LNC�r}A/�_A@�Aο:A�܆@���A�A�}A���An~�@�+2@=SA�A0:�A��&A�j@�G�AV�MAceA-�%A���A�w�Am\tAZw�A�[�A�}<A��7@K�@��cAT��A5$rABگA���A���A��<@���A� �@��B;A���@�jA�~K>�N�A��?AYQ@�  A\��A��A���A���A�~�A��%?rӥ@[�9A���A���A�nV@�oA�A��A��A�{�C���B��A䋊A��PBIOB�-A�~Ay�B	?�A�)sC��mA0��A"�A��qA���@�      
   ,      
                  (         r         
         6      %                              l               !            
   	               $   $      
             '         $                           $   ,                        )                                 9   +               %                     %               7            '      %                              )            #         -                                                                                             '   +                                    !               +                  !                              )            #         '                                                   O$gLN�+�OpcN���N��N��N��Oq(cN�*M矫Og��O��NJ��O��P��OT�HOT O@�}NNO��dO	�O��N��Oi�rN�$\NM,O�1�N�mN[��M���O[��P(V7N�,N^�N�ۭOm	�N�}�O��;N8�O5P�N��N�XOAޖO2,O,�N!�LO#��O�2�N��+N��O��O���O�gO��eO�3�O��sO�Z�N?��N�	�Oy;`O?�O@�JO?MNA�:Nk�HO���OW��N��N�PN��}N��N1�x  �  �  +  �  3  i  �    �  9  �  �  �  
[  x  g  �    �    m  \  B    �  N  r  D  ^  v  ~  �  �  �  �  \  �  �  �    [  �    R    ,  E  �    j     V  +  ?  �  V  �  x  �    i  �  �  %  �  '  	�  �  ?  �  �  �;o;o�u�D����o�ě���`B�t��t��e`B���㼬1��o�aG���o��o��o��j��t��\)���㼴9X���㼛�㼣�
���
���ͼ�9X������h���L�ͼ�������P�0 ŽC���P��w�,1�,1�8Q�8Q�8Q�8Q�T���@��H�9�H�9�T���T���T���ixսaG��m�h�u�y�#��%�y�#��+�����������O߽�\)��{�������T��1������`Bttu�������������~xvt*59BJN[\b`[NB<5/****��������������������NO[dhtuvutplh[YOOLJN;<AIUbdba[UPJI??<;;;jnsz��������zronjjjj��������������������s��������������tpmks���������������;BOQVPONB?;;;;;;;;;;��������������������ghnt���������ztlhccg��������������������6<IUbjrttibUIB<54446[gt����������~o[NOS[��������

��������������������������"*26COTYZVQHC6* ��			������������������������������HOUanz������ztnaUPHH��'-,))%������������������������������

������Z[fgtv���}utg[ZZZZZZW[ghnlg[SSWWWWWWWWWW"/HTXaimrutmaTKF:/'"����������������������������������������)),+-)))@O[bhiighihe[OGDB;9@����&(!"�������V[dgkottuttge[ZVTTVVstz�������ttssssssss46ABINOROOB652/04444LN[cgty�����tgf[SMLLNO[hlmlloh[ROJIJNNNN?I]b{��������{bU<46?P[gtxxtg[QPPPPPPPPPP��#03020#
���������	���������������� ������������)6:?<62)������������������������������

������������������������������������������������������������������������������������������������������������������y|}���������������{y�����	���������Q`t���������tpgb[WOQy����������������zty|���������������}}}|#/<HOSa[YVNH</-#����������������������������������������;ADDHUanuxunaUH<504;�����������������w|�������������zvuw
#+<?<963/#
	QUalmbaUTJQQQQQQQQQQ������������������������������������������
!#/63/-$
��������������������
 ����558BENPTTTNB;6545555tt��������{tjmttttttaY^anz{zvnaaaaaaaaaa�ݿؿѿʿ˿Ϳѿٿݿ���������������g�\�Z�X�Z�\�e�g�s�����������y�s�g�g�g�g�����������
�#�/�<�?�E�J�L�M�H�<�/���m�k�d�`�^�]�`�m�y�{�����������������y�m�ʼż����������Ǽʼּ߼�����߼ּʼʺֺԺɺĺ������ɺֺ�������ٺֺֺֺֿ��ݿؿٿݿ����������������������߽ٽݽ������!�%�$� ����������������������� � ���������a�]�_�a�n�z�~�z�q�n�a�a�a�a�a�a�a�a�a�a��	����!�-�:�F�S�_�l�x�z�l�_�S�-�!����׾ѾоԾ׾����
�����	������.�)�-�.�2�;�D�G�I�G�F�;�.�.�.�.�.�.�.�.�����������н����(�:�J�K�A�4���齫�����������������	��"�)��'�/�H�J�;�/�	����������������������������������������y�s�m�h�e�m�y�����������������������	�����߾پ޾����	��"�'�.�+�"���	�	������������	�����	�	�	�	�	�	�	�	�H�;�4�0�.�0�8�;�H�T�a�m�z��������|�a�H��������������������������������� ���������ֺӺɺǺʺֺ���#�-�6�5�2�,�!���лŻû������ûĻлܻ�����ܻлллоʾþ������������ʾ׾����������׾ʾ���������(�4�6�:�8�4�(�����Z�Q�R�Z�f�s�v�|�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���������������������	�������������������������	���������������������������������������������лϻû»ûлܻ���ܻܻллллллл�������������������������������������������r�f�L�3�0�@�M�����������������������u�l�u�xƁƎƚƛƧƳƷƶƳƨƧƚƎƁ�u�u����¿º¿�����������������������������˻S�Q�S�S�_�l�m�x�����������x�l�_�S�S�S�S��������������������	��%�+�)�������ù������ùϹܹ��� �������ܹϹùùù������s�q�a�_�q�������������������������������������	�
���	������������������������������������$�'�-�-������	����������	�� �"�-�.�"��	�	�	�	�	�	ā�{āĄċĈčĚĦħĳĴĳįĦĚčĈāā�;�9�5�1�3�8�;�H�T�]�a�f�l�p�p�m�a�T�H�;���������������������������������������	���*�+�6�8�@�>�6�*�����������������������������������������Ѻ���������������'�/�3�6�8�5�3�'��������úĺɺ�����-�:�S�N�:�!���ֺ��T�R�N�S�T�`�a�m�t�y�t�m�a�Z�T�T�T�T�T�T������������������������������������������ ������"�(�-�5�8�=�8�/�*�(����@�4�'�����8�M�Y�f�r�����|�r�g�Y�M�@�������������ʼ�����	��������ּ�����ĳĦĝėĖęĦĳĿ������������������Ŀĳ���������������
��0�G�b�n�l�b�<�0�ŭţŠŔŋŕŠŭŹ��������������������ŭE�E�E�E�E�E�E�E�E�FFF$F1F=FFFAF=F(E�E��������������������������������������������ĿĳĿ�������������������������N�5�(�����&�5�A�N�\�c�e�j�v�s�g�Z�N�I�E�=�9�;�=�G�I�V�b�o�y�{�}�{�u�o�b�V�I����ƳƧƚƎƆƔƚƧƳ�����������������̿�ݿѿοϿѿؿݿ��������������Ŀ¿��Ŀѿݿ޿ݿݿѿĿĿĿĿĿĿĿĿĿ����������
�����$�$�$�������������6�)������)�6�B�O�[�h�t�w�h�`�T�B�6ED�D�D�D�D�EEEE%E*E7ECEDE>E7E*EEE������������������������`�X�S�Q�R�S�_�`�l�y�������������|�l�d�`ùõìæìôù������������������ùùùù���
���#�/�<�=�=�<�7�/�#��������ܻлͻǻɻлӻܻ�����������  E ( 7 F I 3 2 I Q K  M c j 7 S 1 { ' z D . & . @ f Z # f ^ F d * G * : G e _ 0 Z ? 7 ; b H g M < � \ F . ` - 0 S R I + q & : n 4 ! J ] M * L  _    �    �  �  �  �  �  /    T  x  �  �  �  5  �  D  �  �  I  �  �  �  S  �  S  b  "    D    n  �  �      �  �  �  �  �  X  x  -    ]  �  �  �  ~  Y    �  #  �  z  �    >  �  �  K  �  G  �  V  ?  �  �  _  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  
  K  e  s  ~  �  �  �  z  e  I  $  �  �  �  \    �  9  �  �  �  �  �  �  �  �  �  �  �  �  k  Q  5    �  �  �  0   �  h  T  n  �      &  *  )    �  �  u    �  �  T  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  j  X  E  2    3  3  3  ,  %        �  �  �  �  �  �  �  k  L  +    �  i  _  T  A  -      �  �  �  �  �  i  K  ,      <    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  u  Y  @  "    �  �  �  _  =  �  �  �  �  �  �  �  �  �  �  �  �  }  h  L  +  �  q  [  N  9  =  B  F  E  C  A  >  :  6  1  +  $  
  �  �  E    �  �  �  �  �  �  �  �    b  @    �  �  �  E     �  c  �  )  T  q  �  �  �  �  �  �  �  �  |  _  :    �  �  {  G  	  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  ^  L  ;  )  	&  	@  	  	�  
  
D  
X  
Z  
Q  
E  
<  
*  	�  	�  	  y  �  �  j  �  x  e  O  2          �  �  �  �  �  z  ]  5    �  �  I  g  T  c  a  d  e  W  @  !  �  �  �  u  c  F    �  }  9    �  �  �  �  �  �  �  s  `  G  *    �  �  �  �  �  �  x  h  �  �                �  �  �  �  m  ?     �  M  �  "  �  �  �  �  �  �  �    y  r  k  d  ^  V  O  G  ?  7  /  '  @  �  �  �  �      �  �  �  �  �  w  0  �  T  �    1  .  m  \  K  ;  ,    
  �  �  �  �  �  y  d  P  =  )        X  [  [  I  (  �  �  �  I    �    :  �  �  �  S    �  �  B  .    �  �  �  �  �    l  M  "  �  �  �  �  �  c  )  �    �  �  �  �  �  e  G  ,    �  �  �  �  u  <  �  �  A   �  �  �  �  �  �  �  �  �  �  �  �  r  e  X  J  I  K  M  P  R  N  F  >  7  .  $        �  �  �  �  �  �    h  R  ;  $  c  l  p  r  q  m  b  S  @  &    �  �  ~  \  7  �  �  �  N  D  B  @  >  <  :  8  5  3  1  2  5  8  <  ?  B  E  I  L  O  ^  S  H  =  4  *        �  �  �  �  �  �  �  u  l  h  d  `  c  e  h  k  n  q  r  s  s  t  u  u  \    �  p  "  �  �  =  a  x  |  o  Y  ?    �  �  �  �  �  �  ~  w  l  ^  U  �  x  �  �  �  �  �  �  �  h    
�  
#  	�  	  L  �  �  �  �    �  �  �  �  �  �  �  v  _  F  *    �  �  �  �  g  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  W  <    �  �  i  �  m  �    �    F  U  Z  \  Z  R  I  >  /      �  �  �  V  *            D  �  �  �  �  �  �  �  a    i  �  �  �    @   b  x  �  �  {  k  N  .    (    	  �  �  �  �  w  I  �  l   �  �  �  �  �  �  �  �  �  �  �  �  z  p  e  [  K  8  &        �  �  �  �  �  �  �  �  &  >  ?  9  *  �  �  �  j  G  "  [  Y  W  P  H  ;  ,    
  �  �  �  �  �  �  �  }  m  c  Y  �  �  r  t  �  �  t  `  K  5       �  �  �  �    X  ,               �  �  �  �  �  X  )  �  �  c    �  N  �  !  R  Q  N  F  :  ,    	  �  �  �  �  �  q  P  '  �  �  �  [    �  �  �  �  �  �  �  n  R  4    �  �  �  �  �  ^  5    ,    �  �  �  �  �  q  V  =  $    �  �  �  �  �  �  �  �    4  ?  D  A  5  !    �  �  �  u  G    �  l    �  L  �  �  �  {  n  Q  (      �  �  �  �  l  ?    �  �    >  R          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  j  i  e  a  Z  Q  C  1      �  �  �  �  v  ]  C  (       �  �  �  �  �  �  �  �  t  A  
  �  �  �  Z    �  �  I  V  J  7  !    �  �  �  �  _  1    �  �  Z     �  �  �  [  +      �  �  �  u  H    �  �  �  T    �  {  "  �  X    #  9  ?  8  -      �  �  �  D     �  X  �  v  �  <  �  {  �  �  �  �  �  e  >    �  �  T    �  �  _  K  #  �  �  \  V  T  M  A  ,    �  �  l  i  s  k  d  Z  I  *  �  �  h    �  �  �  c    �  �  `    �  �  Z  #  �  �  �  K    �  �  x  h  X  H  7  '      �  �  �  �  �  �  �  l  T  <  %    �  �  �  �  �  �  y  Z  ;    �  �  �  �  T  (  �  �  <      �  �  �  �  �  �  �  �  �  �  y  ]  :    �  �  b    j  e  g  h  _  J  1    �  �  �  v  J     �  �  v  0  �  p  �  �  v  e  ^  `  l  �  v  V  -  �  �  �  g  "  �  �    g   �  �  �  �  �  �  w  _  B  #    �  �  �  r  A    �  �  �  @  %          �  �  �  �  �  �  �  �  �    e  K  .    �  �  r  `  O  :  &       �  �  �  �  �  }  q  f  W  7    �  '    �  �  t  7  �  �  m  H  "  �  �    �  7  �  s    �  	�  	�  	�  	�  	�  	�  	i  	0  �  �  R  �  �  &  �  �  )  r  �  0  �  �  �  �  �  �  |  r  g  \  O  A  2  $      �  �  �  �  ?  4  (      �  �  �  �  z  ^  C  (    �  �  t  '  �    �  �  �  �  k  O  2    �  �  �  �  n  L  +    �  �  �  q  �  �  �  �  �  z  o  ]  E  (    �  �  �  W  &  �  �  �  U  �  d  F  (    �  �  �  {  T  +     �  �  r  >     �  o  +