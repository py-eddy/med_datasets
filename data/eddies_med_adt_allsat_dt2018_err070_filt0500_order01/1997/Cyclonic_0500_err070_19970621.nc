CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��O�;dZ       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�	   max       P�
>       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       <��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @E��Q�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vmp��
>     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P�           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @���           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �O�   max       <u       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�^    max       B-�9       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x;   max       B-�        :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =|�[   max       C�GM       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@�_   max       C�B�       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ]       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�	   max       PJ+�       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?ѽ���v       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       <��
       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @E������     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vmp��
>     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�w�           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?Ѽ�n.�     �  \8               	      	         7      	                     "   #            /   B            !   
   (                              5      	      0      \               .   5      	   D                           J   	      	   *N'�$O�w N5��NO��O	�O�M�M��O3��N#b3PM�-N��7O\q�O���N@�7PG.�O�O�8O6%Oc�\O_F�N�P	-�O�:PJ+�P�
>Nb&qN�6TOBW3O�>OĈO�{�O�c�O���O�A�N7�bO�!�O>�O��NO  O�p	P��N�*�N���P
M�O�QNv� P#u�N�אN3�zM�	Nޔ�P��O���No�N���PL�O��;O_��N+��N���N�?O�8?N�3#Ou��P9��O:�JN��N=@OL�<��
<t�;D��:�o���
��`B�o�o�t��#�
�#�
�#�
�D���e`B�e`B�u��o��C���C���C���C���t����
���
��1��1��9X��9X��j��j��������������/��`B�������C��\)�\)�\)�\)�t��t��t���P��P�����#�
�#�
�'0 Ž0 Ž0 Ž8Q�<j�<j�D���D���e`B�e`B�u�u�u���P���P�ě�,/5;AGB;/-,,,,,,,,,,�
#+1::7/#
 ����������������������
#%(#

����������������������������������� ������������

������������)+/006BEPSTOLDBA96*)��������������������[t�����������tgZUTP[��������������������%)-BJOU[hnogbVOB6+'%-5B@@;@[gx��tgNB9*--/<<<9<=<:/*&(------����������������������������������������bghtx���������tgc_bb��������������������:DN[^gotuqjg[NLB=99:����������������������������������������Z_gt�������������c[ZQTamz������zmaYTROOQ������ �����������������)10$����������������������������TT^amowsmleaXTSNOQTT������
���������������������������������������������|�)6;[nrlfh[H6)BIO[ht������}th[MEAB����������������������������������������jmsz�����zmljjjjjjjj��%)2)"��������������������������ht���������������h]h�������������������y|����������������|y��������������������������������������������������?GNgt�������ysg[G:7?HUaknwz���|qaUH@>@HRUannpnjaULNRRRRRRRR2BUa}�����~znaULB872xz�������������zttxx9;BHIJH;149999999999U[cgmmmg[[UUUUUUUUUU��������������������*5Bgt���������gNB5+*����������������������������������������������������������������� ./(����������������������������������������������IIQUbnnpnbUIIIIIIIII-08<GIJIIEA@<:20*(--:<DIKUW^a_YUI?<:98::[gt�����������tg[TU[	
#(06:<=<20.#
		�������������������������#)($�������
)1:<1)
##+/;<=<95/)#  ####Z[cehijlsih[[VZZZZZZ
#2<HSVULH</#
����ĿļĿ������������������������������ŠśũŹ���������� ���������ŹŭŠ���������������������������������������
���!�"�/�1�/�/�"�"���������M�I�A�4�2�/�-�4�9�A�M�O�Z�[�g�i�f�Z�S�M�T�H�T�a�p�~�����������������������m�a�T²°±²·¿������¿²²²²²²²²²²���׾ʾ������������	��"�(�"������s�r�s�}�������������������s�s�s�s�s�s�d�w������������������������������s�d��������������������� �������������������������������������������
��#�%�#��
��������s�Z�M�A�1�2�A�Z�s��������������������������������������������������������y�\�G�D�G�T�`�y�����Կ޿޿����������y�y�o�m�`�V�T�I�T�`�m�y�����������������y�������������������������ĿɿͿʿĿ�������
�����)�6�>�B�K�O�[�O�I�B�6�)�������������������)�-�7�7�6�.�)���ìèàÖÓÏÌÌÓàìù����������þùì����	��	���!�"�#�"��������������߾۾ҾԾ��"�.�;�G�R�G�;��	����ƵƦƠƞƧƼ����������������������������տտ����(�5�A�Z�g�s�}��s�Z�A����������Ŀĳī�t�b�b�māĚĳ�����
�4�A�0�#���U�M�I�G�I�U�[�b�n�t�u�n�b�[�U�U�U�U�U�U���������&�(�5�=�A�G�A�5�(��������׾��������ʾ׾���"�,�/�.�"����������������$�0�=�G�I�D�=�0�$�������s�g�Z�Q�N�G�N�X�Z�g�s�����������������s������g�[�g�����������������������������m�`�\�Z�\�d�g�f�m�z�����������������y�m��	��������	��.�;�I�Y�Z�`�T�G�;�.�������㽺�������Ľӽݽ��#�(�-�'���ìåàØàëìîù��úùìììììììì���������������������*�3�<�6�*������t�r�a�[�W�W�U�[�h�t�{āčĐĚĝĕčā�t�����{�z�|�{�v�u�l�x���������û����������ֺкɺĺĺɺֺٺ����ֺֺֺֺֺֺֺ��z�m�a�T�D�A�A�@�:�:�H�T�m�{�����������z�-�!���������S�l�����������x�_�:�-�"�!������"�&�/�6�6�2�/�"�"�"�"�"�"�Y�M�W�Y�e�e�r�~�����~�{�r�e�Y�Y�Y�Y�Y�Y��������������������5�B�[�c�[�N�B����������������������ù�������ܹϹù����������������������������������������A�5�-�.�A�N�g�s�������������������Z�N�A���������������������ƾʾоʾ��������������������������������������������������������������������������������������������<�;�/�#�#�"�#�*�/�<�F�H�U�Y�`�U�H�=�<�<�a�U�O�@�>�@�M�T�k���������������}�u�m�a�@�'��
���������'�@�M�r�������o�Y�@²®¦¢¦²¼¿����¿µ²²²²²²²²�f�^�\�e�f�r���������������������r�f�f�ּ��������ļ����!�)�-�.�)�"�����ֺ@�6�3�/�3�:�H�Y�r�~�������������r�e�Y�@ŨŠŔŀŇŊŔŠŭŹ����������������ŹŨ�:�9�-�,�)�-�.�:�F�D�@�:�:�:�:�:�:�:�:�:���������������������ĽнսнĽ½����������ݽٽݽ������� �&������������������������������
�� �-�:�9�/�����������������������������0�'�#�!�$�0�7�=�I�b�o�{�}�~�{�o�b�R�=�0�r�Y�M�F�G�Y�r�������ּ���ʼ��мӼʼ��r�������������Ľн����������ݽнĽ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������ùϹйϹϹù�����������E*E#E$E+E7EAECEPE\ErEiEhE_E^EbE\EQECE7E* 0 y Z e - d q b h [ ] > ] f G P C + $ 6 s 4 / N l R [ � ( 3 5 F 1 t - ; J ^ K S > : L i 8 ( C ? J C - Z Y B I ? ^ : n K 3 . l 9 J G ` ~ r    ;  �  B  �  5    M  �  �  �  �  �  
  Z  �  }  )  �  �  �  b  y    �  �  x    V  n  %  ;  (  :  �  C  S  �  �  a  @    �  �  �  �  x    �  N  %  �  �  �  G  �  h  g  �  �  �    �  �    J  �  �  |  <u�t��o�o�T����9X��o���
�T����o�e`B�����h��t��#�
��9X��/�49X�L�ͽT����1�<j�#�
��+��-�ě���h�+�aG��o��+�<j�aG��'C��@��m�h�Y���w�e`B��-�'0 Žy�#��1�0 ž��',1�#�
�q���� Ž�vɽH�9�P�`��G���O߽m�h�P�`�Y��q����u���P�	7L��O߽�Q콧O�A�^ B'�B��BUBؒBDxB�B jB!��B
�yBa�B��B��BX�B+PB�'B	�nB�,B�xB�_B!CBRA�\�B�BBLBƛA�z�B�B7]B�BBZB�B �2B!6�A�iGBi�B�jB�$B#I&BaB�B��B �&B	��BkB$ BfEB��A���B	�B��B	~B ��B�xB+sB-�9B�B��B'�'B&9rB&��B	�.B%D�B��B��B͎B6�B�4BafA�~>B��Bj8B"�B�qB�cB6�B;B!�rB��B4�B�B�tB@B+F�B�xB	�FB�PB�LB��B!HfB
a�A��LB��B��BA�|�B?'B@�B�B�QB�B �B!�_A�~\B6�B�lB �B#@B�]B41B7�B ÓB	�B��B=�B�^B25A�x;B�B@DB
/�B �CB�}B+RSB-� B��B�0B'�B&�B&��B	��B%�lB�5B@�B�BC�B�rB@IA�A���A�	=A�7�A<kXA�'A�?5AV��AG.�A���A���A�k�AB�A�ݥAt��Al�qAu'A�uA��/A�wA�D�A[�MBS�A���A�[�A�<�A�ZAV�;B	m A��A�~�Am:JA_~�A-��A̕�A���A��+@��@<�?A��@�rA���?�o�A��R>��LA���A��AL�A���A��AAìA��@�.�A���@���A{�?�S~A�F�@|t�A#��A0��A��@�O�Bv�@��A*jC�GM=|�[C���A�A��}A��_A���A;�A�ŏA��JAV��AGA�i�AЀ5A���AA��A�_rAv��Al�Au��A��oA�]�ÁxA��A[cBE�A�q�A�|A��3A�s7ASE�B	��A�YA��qAn�VA`�A+0A��A�bA܈v@���@;&9A���@{��A��O?�E�A�o�>��A��7A��$AK��A��AA�{�A�~�A��@�5�A�}}@�%A��?�<A�T�@tb�A"A�A1�A�[@���B�f@���A(B�C�B�>@�_C��               	      	         7      
                     "   $            /   C            !   
   )                              5      	      1      ]               /   5      	   D                           K   
      	   +      !            #            3         '      7                     )      /   A                  )         #            !      %   -         )   !      +               %   '         )   '               !         /                  !            #            '         #      1                     )      /   /                  %         #                  !   -         '         #               #            #   '                        /            N'�$O�w N5��NO��N��+O�M�M��O3��N#b3P�(N��7O\q�O��gN@�7P/�,O�O�8O��O�zON�P	-�OPS�PJ+�P��Nb&qN�6TO*\�Oc7CNߠOO˰�Ol��O}QO}�N7�bO��]O�Ou6[NO  O��0P��N�*�N���O�9lO��sNv� O��lN�אN3�zM�	Nޔ�O��Oh5No�N���O� JO��;OJf�N+��N���N�?O��aN�3#Ou��P9��O:�JN��N=@N��  )  �  �    8  �  �  �  �  �  �  �  �  �  #  �  h  ]  >  �    �  A    i  8  �  �  �    B    q  �  �  �  4  !  �  p  �  [  �    _  �  
�  
      p  0  �      :  H    �  M  �  f  �  �  	N  �  \  j  	�<��
<t�;D��:�o��`B��`B�o�o�t��ě��#�
�#�
�e`B�e`B��o�u��o��1���ͼ��ͼ�C���t���j���
�'�1��9X��j���ě�����`B��h��`B��`B�o�\)�o�C��t��\)�\)�\)����w�t��ixս�P�����#�
�@��P�`�0 Ž0 ŽH�9�8Q�@��<j�D���D���y�#�e`B�u�u�u���P���P��`B,/5;AGB;/-,,,,,,,,,,�
#+1::7/#
 ����������������������
#%(#

����������������������������������� ������������

������������)+/006BEPSTOLDBA96*)��������������������Z`gt�����������tgbZZ��������������������%)-BJOU[hnogbVOB6+'%5FCB>BN[gqz��tf[QB45-/<<<9<=<:/*&(------����������������������������������������bghtx���������tgc_bb��������������������?BDNQ[bgmpngg[NCB>??����������������������������������������Z_gt�������������c[ZSTYamwz����zma\TQQS������ �����������������''!���������������������������TT^amowsmleaXTSNOQTT�������	�����������������������������������������������)7O[fmnh_[O6)LO[ht�������th[PGCDL����������������������������������������jmsz�����zmljjjjjjjj��!)% ���������������������������ghr���������������ig�������������������z~����������������~z��������������������������������������������������DHN[gt�����vog[N@;>DAHU[anz����znaUHEA?ARUannpnjaULNRRRRRRRRDJUan������znaUQLCCDxz�������������zttxx9;BHIJH;149999999999U[cgmmmg[[UUUUUUUUUU��������������������05@Ngt���������gNB30����������������������������������������������������������������++"������������������������������������������������IIQUbnnpnbUIIIIIIIII-08<GIJIIEA@<:20*(--:<DIKUW^a_YUI?<:98::[^gt���������tg[VVW[	
#(06:<=<20.#
		�������������������������#)($�������
)1:<1)
##+/;<=<95/)#  ####Z[cehijlsih[[VZZZZZZ
#&./#
	����ĿļĿ������������������������������ŠśũŹ���������� ���������ŹŭŠ���������������������������������������
���!�"�/�1�/�/�"�"���������A�=�4�2�2�4�A�M�V�Z�b�_�Z�M�A�A�A�A�A�A�T�H�T�a�p�~�����������������������m�a�T²°±²·¿������¿²²²²²²²²²²���׾ʾ������������	��"�(�"������s�r�s�}�������������������s�s�s�s�s�s����������������������������������������������������������� �������������������������������������������
��#�%�#��
������s�Z�M�D�;�5�7�A�M�f�s���������������������������������������������������������y�^�T�K�T�m�y�����Ŀѿڿܿ�����ݿ����y�o�m�`�V�T�I�T�`�m�y�����������������y�������������������������ĿɿͿʿĿ���������
���$�)�4�6�B�G�L�E�B�6�)��������������������)�)�3�2�)�'���àÛÓÓÑÒÓÛàìôù��������ÿùìà����	��	���!�"�#�"��������������߾۾ҾԾ��"�.�;�G�R�G�;��	����ƻƳƫƤƧƩƳ��������������������������տտ����(�5�A�Z�g�s�}��s�Z�A������Ŀĥğđĉ�~�r�rāčĦĳĿ������� ����Ŀ�U�M�I�G�I�U�[�b�n�t�u�n�b�[�U�U�U�U�U�U���������&�(�5�=�A�G�A�5�(��������׾ʾ��������ʾ׾���"�)�#��	����������������$�0�7�=�B�C�>�0�$�����|�s�g�Z�R�N�L�N�Z�Z�s�|���������������������j�g�|�����������������������������`�^�\�^�e�i�h�m�y�����������������y�m�`�.��	������������	��.�;�G�U�W�R�G�;�.�������н������ýнݽ�� �(�,�%���ìåàØàëìîù��úùìììììììì�������������������*�0�9�6�2�*�!����ā�w�t�h�e�[�[�[�d�h�l�tāĊčĕĘďčā�������}�|�~�}�w�x�|���������������������ֺкɺĺĺɺֺٺ����ֺֺֺֺֺֺֺ��z�m�a�T�E�B�B�B�>�H�T�m�y�������������z�-�!���������S�l�����������x�_�:�-�"�!������"�&�/�6�6�2�/�"�"�"�"�"�"�Y�M�W�Y�e�e�r�~�����~�{�r�e�Y�Y�Y�Y�Y�Y�������������������5�B�U�_�[�N�B�)��ù����������������ùϹ��������ܹϹ����������������������������������������N�A�9�7�=�N�Z�g�s�����������������g�Z�N���������������������ƾʾоʾ��������������������������������������������������������������������������������������������<�;�/�#�#�"�#�*�/�<�F�H�U�Y�`�U�H�=�<�<�a�[�T�P�D�B�D�R�T�a�z�����������{�w�m�a�Y�M�4�'���'�4�@�M�Y�f�r�|������r�f�Y²®¦¢¦²¼¿����¿µ²²²²²²²²�f�^�\�e�f�r���������������������r�f�f���������Ǽ����!�*�+�&�������ּ��@�6�3�/�3�:�H�Y�r�~�������������r�e�Y�@ŬŠŔŇŇōŔŠŭŹ����������������ŹŬ�:�9�-�,�)�-�.�:�F�D�@�:�:�:�:�:�:�:�:�:���������������������ĽнսнĽ½����������ݽٽݽ������� �&������������������������������
��!�1�3�(��
�����������������������������0�'�#�!�$�0�7�=�I�b�o�{�}�~�{�o�b�R�=�0�r�Y�M�F�G�Y�r�������ּ���ʼ��мӼʼ��r�������������Ľн����������ݽнĽ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������ùϹйϹϹù�����������E7E.E-E4E7ECEPEXEWEPEGECE7E7E7E7E7E7E7E7 0 y Z e 9 d q b h O ] > W f 9 P C '  2 s 4 , N | R [ � ( . 6 H + t - 8 : T K M > : L h / ( 9 ? J C - Z F B I : ^ : n K 3 3 l 9 J G ` ~ ?    ;  �  B  �  �    M  �  �  �  �  �  �  Z    }  )  >  1  R  b  y  �  �  m  x    
  �    �  �  �  �  C     :  .  a  �    �  �  �  5  x    �  N  %  �  '  �  G  �    g  �  �  �    G  �    J  �  �  |  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  )  !        �  �  �  �  �  �  m  S  8       �   �   �   �  �  �  �  �  t  b  P  8    �  �  �  ~  L  $  �  �  J  �  c  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          &  ,  3  5  6  7  8  :  ;  =  C  I  O  V  \  b    "  *  0  4  8  8  8  1  *        �  �  �  �  �  �  �  �  �  �  �  �  �  q  V  =  )      �  �  �  �  �  �  �  �  �  �  �  t  g  X  ?  '  �  �  �  e  *  �  �  �  `  *  �  �  �  �  �  �  �  �  �  �  s  `  K  3    	  �  �  �  �  U  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  M    �  �  ;  �  >  �  D  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  p  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  W  :  #  ?  Z  �  �  �  �  �  �  }  j  d  �  �  �  p  X  <       	  �  �  �  �  �  �  �  �  �  �  z  p  c  T  D  4  %     �   �   �   �            
  �  �  �  �  �  �  q  C    �  �  X     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  e  S  C  2  h  h  e  ^  S  E  5  $    �  �  �  �  �  �  w  a  H  .    �  A  \  \  R  B  .    �  �  �  I    �  X  �  L  �     �  d  �  	  &  ;  =  1    �  �  �  e  #  �  e  �  /  n  �  �  J  u  �  �  �  �  �  u  K    �  �  e    �  e  �  /  �          �  �  �  �  �  �  �  ~  `  C  %    �  �  {  K    �  �  �  �  �  �  �  �  �  �  �  w  S  /    �  �  n  �  �  )  7  ?  @  =  7  +      �  �  �  g  =    �  �  o  +  �      �  �  �  �  �  �  �  �  �  �  �  �    O  �  ^  �  |  �    @  J  U  d  h  Z  ?    �  �    �  _    �      v  8  -  "        �  �  �  �              %  -  5  =  �  �  �  �  �  �  �  �  �  s  a  M  4    �  �  �  L   �   �  �  �  �  �  �  �  �  w  [  =    �  �  �  �  �  �  �  �  U  �  �  �  �  �  �  �  �  �  �  f  0  �  �  �  <  �  h  �  �         �  �  �  �  �  �  �  �  �  �  {  k  [  H  /  �  �  .  =  A  5  %    �  �  �  �  �  �  y  B  �  �  k  %  �  �  	      	    �  �  �  �  `  2  �  �  }  A    �  {  
   p  H  h  q  o  i  \  K  7    �  �  �  �  j  =    �  �  ;  �  �  �  �  �  �  �  �  l  Q  4    �  �  �  U      �  �  +  �  �  �  �  �  �  �  �  m  T  2    �  �  t  ?  
   �   �   f  �  �  �  �  �  �  q  V  7    �  �  �  S    �  �  R    �  �       0  -    �  �  �  S    �  �  %  �  �    �  *  �  �  �        �  �  �  �  �  �  V  &  �  �  �  ~  _  =    �  �  �  �  �  �  �  �  �  ~  i  U  B  2  !    �  �  �  �  Z  p  l  g  b  ]  V  I  7     	  �  �  �  �  �  L    �  �  �  �  �  �  �  i  *  �  �  e  E  6    �  l  E     �  D  =  [  N  B  6  6  8  :  4  )      
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  o  f  Z  L  <  +      �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  b  3  �  �  _  �    ^  W  D    �  �      �  �  �  }  :  �  &  V  `  `  T  �  �  �  �  �  w  l  `  S  C  1      �  �  �  �  �  �  �  
  
�  
�  
�  
�  
�  
�  
�  
_  
%  	�  	�  	>  �  I  �  �  �  r  �  
          �  �  �  �  �  �  �  �  �  o  J  !   �   �   �      
    �  �  �  �  �  �  �  �  �  �  x  c  M  7  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    s  g  p  c  R  C  4  "    �  �  �  �  �  |  l  a  V  ?     �  {    ,  (  /    �  �  �  M     �  �  �  X  C    �    n  .  �  o  {  �  �    q  ]  A  #    �  �  �  a    �    U  ^          �  �  �  �  �  �  �  k  P  4    �  �  �  �  r      �  �  �  �  �  �  �  �  i  S  =  )      �  �  �  l  
  /  :  5  (    �  �  �  _    �  �  9  �  b  �  @  8  "  H  +  	  �  �  r  3  �  �  �  �  �  �  g  <    �  `  �   �            �  �  �  �  �  v  ]  2    �  �  u  W  G  A  �  �  �  u  k  c  [  S  R  [  d  m  t  z    �  �  �  �  �  M  >  /         �  �  �  �  �  �  �    l  Y  E  0      �  �  �  �  �  �  �  �  �  �  m  M  *  �  �  �  W     �   �  Z  W  Y  c  e  _  O  6    �  �  �  6  �  �    |  �  �  T  �  �  �  �  �  �  �  �  �  �  �  �  m  Y  E  1    	   �   �  �  �  �  �  �  �  ~  \  7    �  �  �  �  e  E     �  �  �  	N  	)  	  	  	  �  �  (        �  z    �    Y  �  �  w  �  �  �  �  �  �  �  �  ~  a  B  "     �  �  �  �  �  m  B  \  I  4    �  �  �  �  q  D    �  �  b  �  C  �  �  4  �  j  �  �  �  �  �  �      �  �  �  �  |  `  C  &  	  �  �  F  �  �  �  	�  	�  	�  	�  	�  	�  	{  	7  �  y    �  �  �  8  �