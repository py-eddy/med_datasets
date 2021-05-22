CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�333333      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�+   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <ě�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?������   max       @F\(��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vg�
=p�     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @��          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��l�   max       <��
      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��    max       B5'�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|�   max       B4Ǝ      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >lqc   max       C�h      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >ZEk   max       C�%k      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          L      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�+   max       P���      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�KƧ   max       ?�($xG      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <ě�      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @F\(��     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vg��Q�     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q�           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @��          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��	� �   max       ?�($xG       T�               K   $         
         E            ;      2                  
   !   
                              	                                                      
                     
N6��N�^VN��OB��P�5VP[��O!E�PG�|N��)Oi@#N��P���N4O0|N�Pr�cN�$�P���OQtO��N�prN]�aO��O��Out�N�01O&��N�(NA&�O�uNR/N��0O��O�l=NjVmOC�O�5�Oz�ODŬNCS�O�ıN`uN?�N\��O�S~N���N%�9N�� O�/OI�gO;9�N�+N�G3OL�KN���O{�&N���O�N_�N��dNK��<ě�<�t����
�ě��t��#�
�#�
�#�
�49X�T���u�u��o��o��o��C���C���C����
��1��1��1��9X��9X��9X��j��j��j���ͼ�`B��`B��`B��h�����t��t���P��P�#�
�,1�,1�0 Ž<j�<j�L�ͽL�ͽL�ͽT���Y��ixսixսu��������O߽�����㽰 Ž� Ž� �����������������;<IU[bnpnmgbUIC<;;;;��������������������oz{��������������zoo������*76%�������,5LX[gt�������t[51(,�
#'/<HPNH<#
���)6BOhn��xo_.	LO[dhiotxthf[SOLLLLLBHO[ht���{tpkkhc[OEB#/66/#�
0UbkkhW<0#
���#/<?<;3/#"#*-/<A?C?D<5/#!�����������������������������������}�����������������}}��#<b������vI<#�����������������������Vamz���������mfaYTSV������������}������

#-#"#%#






~z|�����������znrw~HUa�����������znZ]HH�����������������������
������������������������������������������������/BEN[gt������tgNC8./�������������������������������������������������������������������������������������� �������������@BLNPV[aghjng[NB@>>@����
���������aknz������naUPC?CHUa6>FHUadidd``]UH<7326zz�����������zzzzzzz5BOZ`ht�����v[B@3115����������������������������������������25BHN[d[UNGB54222222glt������������tgbgg����������������������������������������LOU[hjsnha[WOKLLLLLL��5BG@;1)������
#*08;10%#
��HHUagnty����ya]UOJGH��������������������:<FHSTKH<256::::::::���		���������
������fkt�����������ytlhef#/2<EA<6/)# ��������������������rty������{trrrrrrrrr���


������������

��������	������'���������������������������������ʼϼѼμʼ����������H�G�@�C�H�U�Y�a�g�n�z�s�r�n�c�a�Z�U�H�H����������������������������
��������f�e�q�{���ʼ����!�:�L�I�.����ʼ����y�m�`�P�R�`�b�Y�y���ĿտۿڿѿȿĿ���¯¯±µ¿��������������������������¿¯���y�m�;�-�*�,�;�T�y���������ѿ�ܿĿ����e�_�\�Y�U�Y�e�l�r�v�}�~���~�y�r�e�e�e�e��������|�y����������������������������6�3�-�2�6�B�E�D�B�7�6�6�6�6�6�6�6�6�6�6�����v�q�y�������(�F�F�=�E�A�4����Ľ��M�I�K�M�M�R�Z�f�h�g�f�Z�M�M�M�M�M�M�M�M�<�4�/�#���#�/�<�H�U�Z�a�f�d�a�U�H�<�<���	�����������	�������������������������	�/�H�a�t�p�a�;�"��	����Z�O�M�A�7�4�(�&� �(�4�A�A�M�V�Z�^�`�Z�Z�������u�a�E�1��(�Z�s������������������Y�L�H�D�A�>�@�L�^�e�r�~���}�}�}�{�r�e�Y�g�C�-��5�A�]�s�����������������������g����޿�������������������;�:�/�"���"�.�/�3�;�=�B�<�;�;�;�;�;�;�O�B�6�-�(�(�+�6�B�O�hāčģĠĚčā�t�O�������������������	�"�/�H�Q�G�;��	����ù÷ëàÙÓÑËÊËÓàìù����������ù�����������������������������������������Y�M�@�>�@�A�J�M�Y�f�r�u���������r�f�Y�f�c�Z�V�P�T�Z�f�s�u�����������v�s�f�f���������ʾӾ׾�׾ʾ��������������������ʾ����¾̾ھ޾�����"�(�*�'���	��ʾ��������	�����	�������������������������������������¾þ��������������������������y�k�`�e�d�m�y�������Ŀտڿ׿ѿĿ���ֺȺƺɺӺֺ˺ֺ����'�-�<�>�-�����ŇņŇŎŔřŠŨũŨŠŔŇŇŇŇŇŇŇŇ������������������������������������ż��0�4�@�M�f�r��������}�r�f�M�@�4�'������������Ŀѿտؿݿ���ݿѿĿ����������ʾƾ������ʾ׾����	�����	���𾱾��������������������ľ����������������_�P�K�S�_�k�s�u�x�����������������x�l�_�l�j�c�a�c�l�p�x�z�|���x�l�l�l�l�l�l�l�l�
���
��#�,�-�#��
�
�
�
�
�
�
�
�
�
����������������������������������������ŭŠŘŔőŐśŠŹ������������������Źŭ�������(�5�A�K�H�A�7�5�(������g�a�Z�U�Z�g�s�������~�s�g�g�g�g�g�g�g�g�ù����������ùϹعܹ�ܹչϹùùùùù��z�n�j�e�]�^�d�zÓàìòù��ùìàÓÇ�z���������������ûлѻܻݻ߻޻ܻлû��������������ùϹܹ���� �������ܹϹù��3�.�'�$�'�3�9�@�F�B�@�5�3�3�3�3�3�3�3�3D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��`�S�G�=�G�K�S�`�y�����������������y�l�`�!��������!�.�:�=�:�:�3�.�!�!�!�!��¿²ª¬´¿������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������������������� �
��
�
����������ĿķĳĲĲĳĶĿ��������ĿĿĿĿĿĿĿĿD�D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�E*EE*E7E@ECEPEUE\EdE\EPECE7E*E*E*E*E*E* S i o ] R 5  H 8 F g ; ; O � 1 v ] B p 4 P \ _ g _ 1 2 [ Y & < : 1 U e Q R Y N U u A r  H : * 6 & � R A n 6 * D " Z 0 �    n  �    �  }  o  �  �  �    O  w  ^  -  �  �  
  ]  �  �  �  u  &  �  O  P  g  �  N  D  ]  �  �  �  �  t  s    �  i  �  �  S  �  4  �  A  �  �  �  A  /  �  $  �  �  �  #  ;  �  �<��
<e`B�D����/��1�<j�ě��t���1�C���1��1��1��`B��1������j��+�+�D����`B��`B�L�ͽo�aG��+�0 ż�h��h�m�h�o�\)�'m�h�\)�49X�aG��aG��T���,1��hs�@��@��D���}�}�Y���C����㽏\)���w��%��hs������hs��E���-�� Ž�^5��l��\B�cB'7�B!��B�~B-R=B�qBфB(�BmB��B�VB%�B&�B%B!m�B�3BjxB&l5B!�FA�� B ��BM�B:�B7WB�SBHeB �5B"�B�yB	nB�aB5'�B+i�Bd�BB��B�VB�GBL_B�Bd]B!m�B�9B��B
�jB<�B�6BV�B�B$�nB'<BG?B��BʠB�!B
h?BB��B
;�BAB��B��B'/LB!� B�\B-E�B�1B@Bx+B?�BN=BįB%��B*hB5B!@�B?BȚB&�B"9TA�|�B B@5BC�B:7B��B?!B �rB"�B@�B	C�B:�B4ƎB+�B��BńBB3B��BZBA�B�dB�hB!KSB��B�B
הB?�B��BC=B��B$��B�B>�B�9BA�B¶B
@oB?B��B
>-B?�B"�A��>@�]�A�$:A���A�AscvA��{AoT�?�Z�A�d Aת�A*2nA>jeA��0A���A�
�A;fA��;?�k:A�քA��mA�Z�Aٶ�A��BA�A�A��p@�C�AB@qAP�=AX1@AZ�hALK{As)@R�A���A���@��PAy�AY@AL d@��=@�v�A��A�JKA��aA��QA�>lqcA�U@� y>���?��iC��A�A#A�mnC�hA��A�pRC�N?C��XA��@���A�5&A�uEA�cAs��A��Al��?��IA�z)AׁA$�vA>��Aē�A��tA��rA9\�A�S�?��A�gEA��0A��Aؒ�A��	A�}�A�t@��:AA�hAR��AZM�AZ��AL�zAu�@K��A�A�m�@���Aze�AY2	AL��@�SO@�4�A�<�A�-3A�;TA��#A���>ZEkAʁ;@�d�>�P�?��C��jA��A;A�CC�%kA���A�~9C�H�C���               L   %                  F            ;      2                     "                                 	                                                                           
               ?   1      3            7            1      A      '         %   '                  '            '                                                                                                9         3            1                  A                  '                  !                                                                                             N6��N�^VN��N�B�Pn�"O���OV,PG�|N��)OFhkN��Pd�N4N��N�O�=N�$�P���OQtOZ�N�prN]�aOa�UO��O�N�y!Np�"N�(NA&�O��kNR/N��0O��O�� NjVmOC�O�5�Oz�N��NCS�O�ıN`uN?�N\��O:�N���N%�9NM�*O�m�O"�N�g�N�+N�G3OL�KN���Oi`�NeR�O�N_�N��dNK��  �  �  t  ^  �  l  f  v  G    �  �    E  �  �  2  �  A  t  �  �  �  �  S  �  ~  !  �  �  %  .  �  =  �  �  �     �    �    �  J  �  T  �  @  �  �  �  �  �  �    P    �  p  	4  �<ě�<�t����
�e`B��o�����49X�#�
�49X�u�u�ě���o���㼃o�#�
��C���C����
��h��1��1��h��9X�o�ě��o��j���ͽC���`B��`B��h�+���t��t���P�0 Ž#�
�,1�,1�0 Ž<j�D���L�ͽL�ͽ]/�]/�aG��}�ixսu��������\)�������㽰 Ž� Ž� �����������������;<IU[bnpnmgbUIC<;;;;�����������������������������������������������	/1�������45;BN[g����tgYNC<64�#&/5<HNLH<#
���)6BOhn��xo_.	LO[dhiotxthf[SOLLLLLJO[`ht|�~ytmh[ROLGDJ#/66/#0<UdghdSI<0# #/<?<;3/#" #/;<<><2/%#      ����������������������������  ����������}�����������������}}��#<b������vI<#�����������������������acmwz{~����zma[WVWZa������������}������

#-#"#%#






��������������������HUa�����������znZ]HH�������������������������
������������������������������������������������58BN\gt�����tg[L?945�������������������������������������������������������������������������������������� �������������@BLNPV[aghjng[NB@>>@����
���������aknz������naUPC?CHUa8<<HUYZVUH<888888888zz�����������zzzzzzz5BOZ`ht�����v[B@3115����������������������������������������25BHN[d[UNGB54222222fjjpt������������tgf����������������������������������������NOY[dhokh[ONNNNNNNNN��5BE>8,)������

#',/0.#
  
NUZaeknprvnaaUUQMINN��������������������:<FHSTKH<256::::::::���		���������
������glu�����������{tmifg#//;<><3/,#!��������������������rty������{trrrrrrrrr���


������������

��������	������'���������������������������������ʼϼѼμʼ����������H�G�@�C�H�U�Y�a�g�n�z�s�r�n�c�a�Z�U�H�H�����������������������������������꼽����m�u����ʼ��!�:�G�C�.�����ּ�����������������������ÿƿȿĿ���������°³·¿����������������������������¿°���y�m�;�-�*�,�;�T�y���������ѿ�ܿĿ����e�_�\�Y�U�Y�e�l�r�v�}�~���~�y�r�e�e�e�e���������~�|�����������������������������6�3�-�2�6�B�E�D�B�7�6�6�6�6�6�6�6�6�6�6���~�v�x���������(�6�?�<�2�0�(���Ľ����M�I�K�M�M�R�Z�f�h�g�f�Z�M�M�M�M�M�M�M�M�<�7�/�<�C�H�U�W�a�d�a�`�U�H�<�<�<�<�<�<���	�����������	�����������/�"�
����	��"�/�;�H�T�X�c�i�e�]�M�H�/�Z�O�M�A�7�4�(�&� �(�4�A�A�M�V�Z�^�`�Z�Z�������u�a�E�1��(�Z�s������������������Y�L�H�D�A�>�@�L�^�e�r�~���}�}�}�{�r�e�Y�P�N�<�A�G�N�Z�s�������������������s�g�P����޿�������������������;�:�/�"���"�.�/�3�;�=�B�<�;�;�;�;�;�;�B�8�6�/�-�.�3�6�B�O�Z�h�l�tĀ�t�h�[�O�B�������������������	�"�/�H�Q�G�;��	����àà×ÓÔÔàìòùþ����üùìàààà�����������������������������������������f�b�Y�M�L�K�M�Y�Y�[�f�o�r�k�f�f�f�f�f�f�f�c�Z�V�P�T�Z�f�s�u�����������v�s�f�f���������ʾӾ׾�׾ʾ���������������������׾ɾʾ׾�����	���#�!���	�������������	�����	�������������������������������������¾þ��������������������������y�k�`�e�d�m�y�������Ŀտڿ׿ѿĿ��ֺ˺ɺͺٺ������'�1�-�(�!�������ŇņŇŎŔřŠŨũŨŠŔŇŇŇŇŇŇŇŇ������������������������������������ż��0�4�@�M�f�r��������}�r�f�M�@�4�'������������Ŀѿտؿݿ���ݿѿĿ�����������������	����	���������������������������������������ľ����������������_�P�K�S�_�k�s�u�x�����������������x�l�_�l�j�c�a�c�l�p�x�z�|���x�l�l�l�l�l�l�l�l�
���
��#�,�-�#��
�
�
�
�
�
�
�
�
�
������������������������������������������ŹŭŠŜŗŔŒŝŠŹ�������������������������(�5�A�K�H�A�7�5�(������g�a�Z�U�Z�g�s�������~�s�g�g�g�g�g�g�g�g�ù¹��������ùϹԹ۹ѹϹùùùùùùù�Ç�z�l�g�_�a�l�zÓàìïùþùõìàÓÇ�����������������ûлۻܻܻۻлû����������������ùϹ׹ܹ�����������ܹϹù����3�.�'�$�'�3�9�@�F�B�@�5�3�3�3�3�3�3�3�3D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��`�S�G�=�G�K�S�`�y�����������������y�l�`�!��������!�.�:�=�:�:�3�.�!�!�!�!��¿²«®¶¿������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������������������� �
��
�
����������ĿķĳĲĲĳĶĿ��������ĿĿĿĿĿĿĿĿD�D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�E*EE*E7E@ECEPEUE\EdE\EPECE7E*E*E*E*E*E* S i o f O F � H 8 7 g 6 ; & �  v ] B ^ 4 P - _ Q _ < 2 [ P & < : 5 U e Q R 0 N U u A r  H : * 6 + f R A n 6 + B " Z 0 �    n  �    �    M  �  �  �  �  O  �  ^  �  �  u  
  ]  �  /  �  u  �  �  H  0  �  �  N  �  ]  �  �  z  �  t  s    �  i  �  �  S  �  �  �  A  ]  �  `    /  �  $  �  �  �  #  ;  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  D  $     �  t  j  `  U  H  ;  .  !        �  �  �  �  �  �  �  �  �  �  �  �  
  %  B  N  [  ^  W  E  &  �  �  h  C  k  w  f  S  �  �  �  �  �  �  X    �  �  J  �  �  2  �  Y  �  >  P    V  j  �  �    4  I  ^  k  c  H    �  �  |  P    �  U  �  f  f  _  O  5      �  �  �  �  �  n  T  (  �  �  U    �  v  o  g  Y  E  ,    �  �  �  p  Q  E  L  J  9    �  l    G  6  $    �  �  �  �  �  ~  ^  '  �  �  �  I  �  �  W    �  �    �  �  �  �  �  �  �  }  p  P  (  �  �  �  U      �  �  �  �  �  �  �  y  q  j  ]  L  :    �  �  �  R    �  �  �  �  �  �  �  �  �  ?  �  �  J  �  w    �  \  �  �  �            �  �  �  �  �  �  �  �  k  I  &  �  �  C   �  �  �  �  �    D  =  2  $    �  �  �  �  K    �  �  v  J  �  �  �  �  �  �  �  �  �  �  o  S  @  1  #      �  �  �  �    �    �  E  �  �  �  �  �  �  �  #  �  (  �     �  �  2  %         �  �  �  �  �  �  �  u  `  V  L  I  b  {  �  �  �  �  Z    �  �  s  M  %  �  �  �  \    �  t     �    A  1      �  �  �  �  �  �  s  l  g  a  T  C  -    �  �  s  m  b  P  O  m  k  L  )    �  �  �  h  >    �  P  �  V  �  �  �  �  �  �  �  �  y  h  W  F  4  #    �  �  �  Z  (  �  �  |  l  Z  I  5     
  �  �  �  �  �  l  G  "   �   �   �  '  J  �  �  �  �  �  �  �  �  �  �  V  �  �  6  �  O  V  3  �  �  �  �  z  d  M  2    �  �  �  v  X  P  R  6     �   �  "  >  F  M  R  G  M  5    �  �  �  r  6  �    �  .  �  �  �  �  �  �  �  w  d  J  *    �  o  6    �  �  �  �  h  G  �  �  �  �    8  X  o  |  {  o  Y  ;    �  �  �  �  S  �  !          �  �  �  �  �  �  �  �  �    o  _  M  <  +  �  �  �  �  �  �  �  �  �  �  �  �  r  a  P  3     �   �   �    d  �  �  �  �  �  ~  v  l  ^  N  >  #  �  �  C  �  R  �  %           
    �  �  �  �  �  �  �  �  �  X  '   �   �  .  )  %        	  �  �  �  �  �  �  �  �  �  �  ]  "   �  �  y  o  e  X  K  ;  (    �  �  �  �  �  |  c  M  4  '  #    +  =  8  ,  %  *  ,  *  /  *    
  �  �  �    �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  O  A  5    �  �  �  �  g  8  
  �  �  �  �  x  e  X  g  `  N  6    �  �  �  �  s  M  /  H  8       �  �  �  �  �  �  �  �  �  �  v  ^  :  
  �  �  b  +  >  �  s  g  d  g  a  c  z  �  �  v  h  W  <    �  �  o  G  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  m  s  h  ^  L  3    �  �  �  s  E    �  l  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  m  �  �  �  �  �  �  �  �  �  �  �  �  �  �    9  f  �  �  �  J  G  D  B  ?  <  :  7  4  2  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  R  3    �  �  d     �  �  A  T  I  >  1  #    �  �  �  �  |  Q  "  �  �  ~  :  �  �  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  D  -     �  �    $  5  >  @  :  *    �  �  �  �  �  �  o  ]  K  3    �  �  �  �  �  v  e  R  :    �  �  �  k  :    �  �  7  @  �  �  �  �  �  �  v  d  R  >  )    �  �  �  �  �  b  %  �  �  u  �  �  �  �  �  �  �  �  _  1  �  �  �  F    �  �  �  �  �  �  �  �  �  �  �  k  I  '    �  �  �  m  F     �   �  �  �  �  �  �  p  T  8       �  �  �  �  ~  }  �  �  �  �  �  �  �  �  �  �  �  {  u  m  h  e  n  ~  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �    x  t  q  n  j  f  b  O  P  L  E  =  3  (         �  �  �  ~  P    �  ^    �  �          �  �  �  �  �  �  h  D    �  �  x  B    �  �  �  �  ~  o  k  c  R  @  ,      �  �  �  �  �  f  ?    p  ^  L  :  '    �  �  �  �  �  �  y  e  Q  =  (     �   �  	4  	'  �  �  �  e  *  �  �  M  �  �    �  (  �    �  �    �  |  i  U  A  (    �  �  �  �  `  2    �  �  y  O  *  