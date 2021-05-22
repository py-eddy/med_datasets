CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�ȴ9Xb     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MҮ�   max       P��     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��hs   max       <e`B     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?L�����   max       @F�\(�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @v{
=p��       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @Q�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �bN   max       <D��     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4��     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��`   max       B4ӱ     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C��&     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =� Z   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Y     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MҮ�   max       P��     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�E����   max       ?��
=p��     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       <e`B     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @F�\(�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��fffff    max       @vzz�G�       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��          4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��5�Xy>       cx         Y                                 	            *         /         	      	                  1      
   (      %      	               
            D                  6   @                  ;   '      	      X               
      
      1      
M��;ONfP7�wN�{N�
}N�QN��NřPNLN�Q^O��bNuȱN@�*N���O�S�O=��Nv"�P��N�2[O�� P
y!O��=N���N�oqO��ND.!O�ͻN;A�N�N�&�N��-O�`NP/N���O�
O�c�P=��N�,7N�7>O"5SM���O��N#xTNV-�O�>�O=�\OD�JO���N�P�N���O��N��N��bO��P��N��Of"O�8�O2N��oO���O�O�+Nˍ�N��mO��Nby9O@m5N��N�H�N�WdN	�ON+��MҮ�O�L�OReN���<e`B<t�<t�;ě�;�o;D��;o:�o%   ��o��o���
�ě���`B��`B�o�t��T���u��o��o��o��C���C���t����㼛�㼣�
��1��1��9X��9X��9X��j���ͼ��ͼ��ͼ�/��/�����+�C��\)�\)��w��w�#�
�#�
�',1�,1�0 Ž0 Ž0 Ž0 Ž<j�<j�<j�D���H�9�T���T���Y��]/�]/�e`B�e`B�ixսixսm�h�q���u�����+��+��hs��

������������������������������������
&,153#�������EN[gttvtmgc[WNJBEEEE������������������������������������������������������������#/<BC<73/,##$/<=G<7/%#anz~���������zncaaaaBLM[hot���{~~�t[O>CB36BHOOOOOBA640333333��������������������rt}�����������ytqqpr��������������������15N[git}ug_UNDB@5011��������������������y�����������zqwx�}sy������������~|{GKTamszywy~�zmj]TIGG�������������������������������������)/5<5)(��������������������:FMUamz�����wxmaTH::��������������������GO[t��������tjhcOHFG�����������������������	�����  ��      ����������{���������)/464)�������������������������%)6:BDOPOKB<64)&%%%%��������������������0Obnr{{pnb\I0'bpuou������������nbb����������������������
#()#
�������STahmz{��|ziaVTPIIJS6<IUbcbVUIF<66666666#0<INTTMI>0)#��������������������`gt|�����tkg````````otz�������������smlo�������������GHU\ah�zwpnaUMOIC@G������������������������������������������������������������pu}�������������thnp��

�������������������������������oz�������������zmnmo�����������������zx���������������������
#/<GC<</-#
 �36BO[hkidd`[VOFC80.3-0<IUV__`]XWUIC<:10-QUUanvtnkaUQQQQQQQQQoz�������������~tpgoagt�����������td_\^a����������������������������������������@BNOV[^_[VNLB?<=@@@@"/HTTYZXRH</#������������NTZ[ehkt{���vsh[VRMN��������������������������� ���������&)5?BCB@;50)""��������������������st������tpssssssssss������������������������ 	����������








�������������ż��������������������������
���
��#�/�H�U�[�Y�_�U�R�H�<�/�#��
�Z�;�(�(�"�(�A�N�g�s�����������������s�Z�������������&�(�+�(� ��������	��������*�2�*�(���������t�n�g�\�g�t�t�t�t�t�t�t�t�t�g�b�`�d�e�g�h�s�y��������z�s�l�g�g�g�g�H�B�?�D�H�U�Z�a�n�x�zÆ�z�n�a�U�H�H�H�H��ŻŹŶŹŹ�����������������������������H�@�=�<�4�1�<�H�S�U�U�V�Y�[�X�U�H�H�H�H�;�"�� ����ݾپ����"�;�G�N�o�u�m�T�;�F�B�A�F�R�S�_�l�q�m�l�b�_�S�F�F�F�F�F�F��ùùïù�����������������������������ž�����}������������������ľ����������������������������׾�������ؾξʾ������������y�z���������������������������������������y�m�l�i�m�n�y����������������čăĦěĒĦĿ���
�Q�W�V�M�<�����ĿĚč�����������¿Ŀѿݿ��ݿֿѿĿ�����������ƳƢƖƚƧ������������������������������������������$�0�6�:�<�:�0�$���}�m�n�s�����������ž׾��׾ʾ��������}���}�s�i�g�Z�X�\�g�o�s�~�~���������������U�M�P�U�a�n�zÀÅ�z�n�a�U�U�U�U�U�U�U�U�H�/��	�������/�;�H�T�a�m�r�r�w�p�T�H�������������������������������������������������������
�������
���侌�����������������������������������������������������������������ƾ����������������������������	�	��	����������������ÓÎÓàçìóù��ýùìêàßÓÓÓÓÓ�[�H�;�6�,�6�H�hāčĚĤĨĪĦĚčā�h�[ùòíøù����������ùùùùùùùùùù�/�)�#�#�"�#�-�/�1�<�E�H�N�H�?�<�/�/�/�/�����������!�.�G�S�`�k�b�S�:�!� ������������g�S�P�Z�d�s���������������������(��ݿ������q�l�p�������ĿϿ���7�M�E�(�������������������������!����t�n�g�d�g�j�l�t�x�t�t�t�t�t�t�(�'�!�����(�5�A�B�N�V�Z�[�Z�N�A�5�(�-�'�'�+�-�-�0�:�:�;�:�0�-�-�-�-�-�-�-�-�������������ûлܻ����	�����ܻлû�����������������������������������������ùöïìëìù��������üùùùùùùùù�;�1�.�*�+�1�7�9�G�T�^�`�m�u�v�p�`�T�G�;�^�T�H�T�Y�`�f�m�r�y���������������y�m�^���������������Ŀ̿ѿݿ����ݿĿ������'�$��� �*�,�3�@�M�Y�f�r�y�~�|�o�Y�M�'�ܻٻٻܻ����������&�������ܻ�������������������������������������ŹŭŠŜŎőŜŠŭŹ������������������Ź�0�)�$���$�0�0�2�5�0�0�0�0�0�0�0�0�0�0�V�S�U�S�P�S�V�W�b�i�o�v�{ǀ�{�o�l�b�V�V���������r�x�������ܹ���'�/�(���ù��Y�@�3�1�@�Y�e�~�������0�@�C�!���ݺɺY�a�V�Y�X�T�T�T�W�a�m�p�z�t�n�m�a�a�a�a�a�����z�y�t�q�u�z�������������������������s�q�l�q�h�����������������������������s�������������������Ľнݽ��н����������H�<�;�3�1�;�H�R�T�\�\�T�H�H�H�H�H�H�H�H�#�
���������
�#�/�<�H�U�W�`�`�U�J�H�<�#ĳĬĦĢĠĠīĳ�����
���� ������Ŀĳ������������ĿĮĭĿ�������1�6�0�#��e�c�Y�P�L�L�G�L�Y�e�r�w�v�u�x�r�e�e�e�e������#�*�6�A�C�D�C�;�6�*�����E�E�E�E�E�E�E�E�E�FFF%F/F1F*FE�E�E�E��(�'�(�)�3�5�A�N�N�N�N�A�@�5�(�(�(�(�(�(�ù����������������ùϹع�������ܹϹüf�b�\�f�j�r������������������r�j�f�f���������ʼͼּ������������ּʼü��z�y�n�a�_�Y�a�i�n�s�zÇÐÓÖÓÏÇ�z�zùøøù������������������ùùùùùùù�u�t�p�t�{�����������������������������������������û����������ûл�������������ܻü���#�4�8�@�M�Y�e�t�v�r�k�f�Y�M�4�'��`�V�_�`�l�y���������y�l�`�`�`�`�`�`�`�` T M 7 ( @ I p Y D l p ? s + R d = Q T c * M U * 3 * 8 ' ? 3 � = R 7 V N j e r H � $ : ^ 3 F F ' ] Q - m ] � g p V r U ; & & p R I $ O 5 q f = c > d + > I    &  �    	  �  �    �  p  ?  �  �  ^    6  �  �  �  �  k  }  C    �  
  Q  8  M  �  �  �  �  I  �  �  v  �    �  {  �  �  P  �    �  �  �  �    0  _    L  Q  �  l  �  \  �  �  �  �  �  �    �  �  �  �  �  P  S  1  �  �  �<D���T��������o%   ��o�o�t��o�e`B��o��t��49X�u��t���9X�D���]/��1���}�o��9X�����8Q��`B�,1���ͼ�h��/����hs��`B�+����'y�#���t��'+�L�ͽ��8Q�ixսT����7L��/�<j�Y��y�#�<j�L�ͽƧ����L�ͽu���P�q���P�`��"ѽ�^5��C��}�u�bN�}󶽬1��%�}󶽋C��}󶽍O߽�7L��xս\���TB#�`B��B<�B�zBvPB�eBK
B�BC^B]B��B&�B �#B��B!��B
�B J
B�B
�A��%B0	B{B;IB ��A���B�B��B4��BT�B��B��B��B�wB��BH�B&�B)�Bi�B�A��!B&ڶB%��BW?B	ШB
�/B��B,B �+B"�BC�B	�B��B�B��B��BL�B��B�/B&�)BLoB
�MB
x~B^>B!�GB&NBB�B��B��B+�hB-quB��BB�7B�pB�B�HBSRB#�)B�
B77B�MB�*B�nB��B,�BAB~�BG�B8�B �B��B!�B�'B k�B=vB
�lA��`B< Bl�B?{B �A���B@�B�FB4ӱBA�BWYB ?�B�B��B�B@�B&I�B* �B\�B��A�v�B&�fB%��BE4B	��B
�B��B@dB ��B#gnB�?B
�(B,~BDB��B��B�}B��B�pB'C�Bz�B
��B
A�B�B!ҝB>�B��B�B�B+PLB-��B=KBEZB�1B��B��B<�B@%@�o�A�R�A��0A2��A��!A�,�A���A�:�A�j�A�a=A_�g@�A�
xAJ<AP��A�j�AmףA�
bAy6�BB��AIG�A��A�A�JNA�!A�5vAI��AI�tA�ّA�:1A�t�AͫVA��jA�gA��kA{�TA�S7A���A���@o@��A�"�Aͤ�Af%Al��Axw�@�o�@�57A�s�A���B
�B(�>��@()�A��^A���A�ezA%4lA�hRA�*rA�L�A�i?�a�A�T�C��&A��->z�M@���A��A�aA��GA���@�N\@�(Q@԰�A��@�}aA�fA���A3�A��A� A��SAŰKA��FA�c7Ab@��YA΃ZAIEAM�A��fAm��A��dA{�B4�B	>�AH�	A�}�A�pWA�w�A��A�jAJ��AI.�A���AʜA܀vA�~HA�pRA��A�A�Aw�4A�P�A�s�A�m@s�&@�D�A�i�A�mAf��Am VAz&�@���@��[A���A�X�B	��B@=� Z@��A���A���A��UA$�A�KqA�A�YA��?�W?A��kC��A���>CF@��xA�OAǄ�A��|A�~p@�,�@�LH@��AJ         Y                  	               	            *         /         
      
         	      	   2         )      %      
                           E                  7   @                  ;   (      
      X                     
      1               +                        )                     =         %            %                     !         #   )   9                                                   1   A         #         !      #         #                           !                                       )                     1         !                                             '   1                                                      A         #                        !                                 M��;O[�O�@�N�{N�
}N9�)N��N�ҼNLN�Z�O��bNR��N@�*N��O*c�N�Nv"�PY�N�2[OU�O��N�+�N���N�oqO�L�ND.!OiJ N;A�NYTN�&�N��-O��DNP/Nl�O��OߟPe4N�,7N�7>N��M���OU��N#xTN,��O<h�O=�\OD�JO2��N�P�N��Oxt�N��N��bOLڮP��N��O�GO�8�O2N��oOhu�O�N�Oe�Nˍ�N��mO���Nby9Nq�N��N�H�N�WdN	�ON+��MҮ�O��O4ΓN]B�  �  s  	!  J  j  G  �  �  �     +  �  D  L  /  �  C    V  �  �  1  �  �  y  I  H  �  1  x  �  r  j  i  �  H  �    �  �  g  �  o  �  �  �  �  	]  =  V  W    X  �  �  �  �  }  �  `  ~  c  �  <  "  �  �  7  �  Q  �  l  9  8  �  :  �<e`B;��
�+;ě�;�o:�o;o��o%   ��o��o�ě��ě��o�#�
�49X�t���C��u��t���1��j��C���C���9X���㼴9X���
��j��1��9X��/��9X�ě���P������h��/��/�+���\)�C��t���w��w��w��%�#�
�,1�49X�,1�0 Žm�h�0 Ž0 Ž@��<j�<j�D����+�aG��aG��Y��]/��\)�e`B��O߽ixսixսm�h�q���u�����t���O߽�t���

�������������������������������������

������EN[gttvtmgc[WNJBEEEE������������������������������������������������������������ #/:<><61/%#"    #$/<=G<7/%#fnz��������znjffffffBLM[hot���{~~�t[O>CB46BGNOOOFBB641444444��������������������st����������~trrqss��������������������256BNU[`c\[RNGB:5222��������������������yz��~������������zsy������������~|{GHLTagmpxwwutma_TJHG��������  �����������������������������)/5<5)(��������������������EILPYamz���zmaTHC>>E��������������������KOZt��������th[QOKJK���������������������� ����������  ��      ����������{���������� )-23/)�������������������������&)6BMJB965)'&&&&&&&&��������������������!2IRbnz}znmgZI<0#"!qwry������������{nhq����������������������
#()#
�������QT[amz~zumbaXTOQQQQ6<IUbcbVUIF<66666666#0<IKQPIC<0.'#��������������������dgt�����tlgddddddddrt�������������xtrqr�������������GHU\ah�zwpnaUMOIC@G������������������������������������������������������������rw��������������vtqr��

�������������������������������z�����������zvtrrruz�����������������zx���������������������
#/<BB<:/+#
 36BO[hkidd`[VOFC80.3-0<IUV__`]XWUIC<:10-QUUanvtnkaUQQQQQQQQQstw|��������������{scgt�����������tea^`c����������������������������������������@BNOV[^_[VNLB?<=@@@@#/<HOUURL</#
������������U[ahmtyvth[UUUUUUUUU��������������������������� ���������&)5?BCB@;50)""��������������������st������tpssssssssss����������������������������������
�������������ż��������������������������
�	�
����#�/�<�H�N�T�M�H�<�/�#��
�
�g�Z�L�I�I�N�T�Z�g�s�����������������s�g�������������&�(�+�(� ��������	��������*�2�*�(���������t�r�g�c�g�t�t�t�t�t�t�t�t�t�g�b�`�d�e�g�h�s�y��������z�s�l�g�g�g�g�H�D�C�H�L�U�W�a�n�q�z�}�z�n�a�U�H�H�H�H��ŻŹŶŹŹ�����������������������������H�C�@�<�9�7�<�H�N�U�W�Y�V�U�H�H�H�H�H�H�;�"�� ����ݾپ����"�;�G�N�o�u�m�T�;�F�C�B�F�S�[�_�c�l�l�l�a�_�S�F�F�F�F�F�F��ùùïù�����������������������������ž�����~������������������������������������������������ʾ׾�����׾оʾ������������������������������������������������������y�m�l�i�m�n�y���������������������ĿĳĩĭĨėĦĴĿ���
�G�R�S�F�0������������¿Ŀѿݿ��ݿֿѿĿ�������������ƳƤƘƚƠƧ�����������������������������������������$�0�3�7�9�:�7�0�$�������y�~��������������������������������}�s�i�g�Z�X�\�g�o�s�~�~���������������U�M�P�U�a�n�zÀÅ�z�n�a�U�U�U�U�U�U�U�U�;�/�"��	������/�H�V�_�l�l�i�a�T�H�;�������������������������������������������������������
�����
������侌�����������������������������������������������������������������������������������������������	�	��	����������������ÓÎÓàçìóù��ýùìêàßÓÓÓÓÓ�h�[�K�B�;�B�N�hāčĚğĥĦĢĚčā�t�hùòíøù����������ùùùùùùùùùù�/�*�$�$�/�<�D�H�K�H�>�<�/�/�/�/�/�/�/�/����� �����!�3�:�I�G�C�:�.�!��������l�g�Y�Z�g�s��������������������������ݿ������u�q�w�������Ŀʿ���.�9�8�(��������������������������!����t�n�g�d�g�j�l�t�x�t�t�t�t�t�t�5�0�(�%��"�(�5�<�A�F�N�O�N�M�A�5�5�5�5�-�'�'�+�-�-�0�:�:�;�:�0�-�-�-�-�-�-�-�-�������������ûлܻ����������ܻлû�����������������������������������������ùøñìììù��������úùùùùùùùù�;�8�2�1�;�=�G�T�`�f�m�p�q�m�k�`�X�T�G�;�^�T�H�T�Y�`�f�m�r�y���������������y�m�^���������������Ŀ̿ѿݿ����ݿĿ������@�4�+�(�/�4�@�@�M�Y�\�f�m�s�p�f�b�Y�M�@�ܻٻٻܻ����������&�������ܻ��������������������������������������ŹŭŠŠŘőŔŠŭŹ������������������Ź�0�)�$���$�0�0�2�5�0�0�0�0�0�0�0�0�0�0�V�S�U�S�P�S�V�W�b�i�o�v�{ǀ�{�o�l�b�V�V�����������������׹ܹ�����ܹϹù������Y�@�3�1�@�Y�e�~�������0�@�C�!���ݺɺY�a�V�Y�X�T�T�T�W�a�m�p�z�t�n�m�a�a�a�a�a�����z�z�u�r�w�z�������������������������s�q�l�q�h�����������������������������s�������������������Ľнݽ��н����������H�<�;�3�1�;�H�R�T�\�\�T�H�H�H�H�H�H�H�H�<�/�#���
�� ��
��#�/�<�H�U�W�M�H�<ĳĮĨĥĥĳĿ���������	����������Ŀĳ�#��	�����������������������,�2�0�+�#�e�c�Y�P�L�L�G�L�Y�e�r�w�v�u�x�r�e�e�e�e������#�*�6�A�C�D�C�;�6�*�����E�E�E�E�E�E�E�E�E�E�FFF)F+F%FFE�E�E��(�'�(�)�3�5�A�N�N�N�N�A�@�5�(�(�(�(�(�(�ù����������ùϹѹ۹۹Ϲùùùùùùùüf�b�\�f�j�r������������������r�j�f�f���������ʼͼּ������������ּʼü��z�y�n�a�_�Y�a�i�n�s�zÇÐÓÖÓÏÇ�z�zùøøù������������������ùùùùùùù�u�t�p�t�{�����������������������������������������лû������������ûл�������������ܻм4�)�'���&�4�:�@�M�Y�a�f�r�t�r�h�Y�M�4�`�X�`�b�l�y���������y�l�`�`�`�`�`�`�`�` T E 2 ( @ X p V D j p 1 s . \ H = K T H ' < U * 8 * 2 ' ( 3 � / R : = J f e r E � % : J ) F F  ] R * m ] Z g p T r U ; $ $ s R I  O % q f = c > d  3 >    &  T    	  �  Y    �  p  �  �  k  ^  �  �    �  �  �  �    �    �  �  Q  �  M  i  �  �  Z  I  �  k    l    �  �  �  �  P  e  �  �  �  s  �    �  _    �  Q  �  K  �  \  �  �  �  5  �  �  �  �  z  �  �  �  P  S  1  &  �  m  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  T  i  q  p  k  a  R  @  )    �  �  �  e  !  �  �  �   �  �  �  (  �  0  �  �  �  	  	   	  �  �  B  �  :  �  �  T  \  J  F  B  =  5  .  %        �  �  �  �  �  �  g  J  -    j  \  N  @  2  $      �  �  �  �  �  �  �  �  �  �  f  A  2  7  ;  @  C  F  G  F  E  ?  9  2  +  %          �  �  �  �  �  �  �  z  p  e  Z  O  B  2  "      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    |  w  p  j  f  u  �  �  s  �  �  �  �  �  �  �  �  o  Y  @  #    �  �  r  =    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  @    �  �  _  +  (       �  �  �  �  �  �  �  �  �  �  �  �  z  [  *   �  �  �  �  �  �  �    ^  7    �  �  �  U  "  �  �    �  C  D  <  4  -  $        �  �  �  �  �  �  �  x  e  Q  >  +  <  B  I  J  F  A  :  3  $    �  �  �  �  �  a  :     �   �         '  ,  .  )  !      �  �  �  �  �  �  P     �   Q  �  �  �  �  �  �  �  �  �  �  �  �  �  n  N  -    �     c  C  =  7  0  *  $          �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  t  E    �  �  �  �  _    �  G  �  V  Q  M  H  ?  5  ,  #         �  �  �  �  Y  +   �   �   �  &  ]  �  {  i  [  H  4      �  �  �  c  -  �  �  c    v  �  �  �  �  �  �  �  �  q  A    �  k    �  v  
  �  �  M              �    &  0  ,  &      �  �  �  �  w  ?  �  �  �  �  �  �  �  �  �  �  �  �  z  a  H  .     �   �   �  �  �  �  �  �  p  T  6    �  �  �  f  3    �  �  q  E    Y  h  u  x  v  p  f  X  @  "  �  �  �  c    �  �  P    �  I  -    �  �  �  k  <    �  �  �  �  �  �  �  �  �  �  �  ,  >  G  G  F  D  A  9  -       
  �  �  �  �  <  �  >   �  �  �  �  �  �  �  �  �  �  �  �  �  v  j  ^  Q  D  7  +    �    "  ,  /  .  #      �  �  �  �  Z  .     �  �  �  Z  x  r  k  e  ^  V  N  E  ;  1  %    
  �  �  �  �  �  �  �  �  �  |  r  i  `  W  N  C  7  N  �  �  �  �  �  c  �  �  �  -  l  r  p  d  K     �  �  x  3  �  �  ;  �  7  x  �  �  R  j  d  _  Y  K  :  *      �  �  �  �  �  �  q  Y  =  !    g  h  i  i  h  e  `  Y  P  D  3    �  �  �  �  j  ;    �  �  $  O  u  �  �  �  �  r  H    �  �  �  U    �  2  �  �  D  G  F  E  E  C  ?  8  2  &    �  �  �  �  �  �  �  f  J  y  �  �  �  y  u  g  X  .  �  �  m  %  #    �  �  (  �  ?          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    {  x  s  m  g  b  `  `  �     X  �  �    N  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Q  5    �  �  �  �  d  g  Z  N  A  5  )        �  �  �  �  �  �  �  �  �  ~  o  �  �  �  �  �  �  �  �  �  �  �  �  {  n  c  Z  P  D  4    o  q  r  t  v  v  r  n  j  f  a  [  U  P  J  !  �  �  �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      /  A  �  �  �  �  �  �  �  �  �  �  �  �  u  Y  ,  �  �  V  �  �  �  �  �  �  �  �  �  i  L  +    �  �  c  (  �  �  �  r  :  �  �  �  �  z  `    K  S  0  �  �  v  0  �  �  �  r  �  �  �  Z  �  �  	!  	E  	[  	X  	7  		  �  �  G  �  v  �  �  �  �  �  =  7  2  ,  &      	  �  �  �  �  �  t  c  S  9  �  �  b  Q  T  P  B     �  �  �  �  U  &  �  �  �  �  �  �  �  �  w  ,  S  W  V  M  =  )    �  �  �  �  }  V  '  �  �  =  �  ?      �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  W  X  Q  J  E  D  D  E  F  G  :  *      �  �  �  �    d  H  [  �  3  |  �  �  �  �  |  \  9    �  �  5  �  �  �  �   �  �  �  �  {  7  �  �  M  "  7    �  |  &  �  0  �  �  .  �  �  �  �  �  {  `  G  /      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  i  Q  4    �  �  �  �  �  �  �  <  }  c  J  2      �  �  �  �  t  i  W  6  �  �  e  )  �  w  �  �  r  a  N  =  ,      �  �  �  �  �  b  7  	  �  �  �  `  X  O  G  >  6  .  &        
               	    �  �  &  M  m  {  }  v  g  K  !  �  �  M  �  L  �    b  �  N  `  _  \  Y  S  L  B  ,    �  �  f  &  	  �  h    �  =  v  |  �  �  �  �  |  d  F  $         �  �  �  {  I    �  <  5  .  '  !  !  "      �  �  �  �  �  �  |  h  e  �  l  "      �  �  �  �  �  �  �  �  �  o  X  8    �  �  �  l  �  2  l  �  �  c  0  �  �    �  �  )  
a  	�  �  �  +  �  j  �  �  �  w  \  ?  #    �  �  �  w  U  3    �  �  �  �  �  �  �  �  �  �  �  �    &  6  -    �  �  �  �  l  �    :  �  �  y  i  Z  L  =  4  .  (    	  �  �  �  �  �  �  i  F  Q  J  C  <  2  !    �  �  �  �  �  �  �  v  `  E  )     �  �  �  �  �  �  ~  _  ?    �  �  �  �  k  D    �  B   �   �  l  a  W  L  B  7  -    �  �  �  �  �  ~  c  G  ,     �   �  9  ;  =  0  !    �  �  �  c  4    �  �  m  9    �  �  [  8  ?  F  M  T  [  b  h  o  v  z  z  {  |  |  }  }  ~      �  �  �  �  �  p  <    �  �  w  @    �  [  �  j  �  �  �  :  .  9    �  �  �  �  `  *  �  �  Z  	  �  W  �  �  /  �  �  �  �  �  ~  q  e  Z  N  C  7  *    	  �  �  �  �  �  �