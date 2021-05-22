CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�fJ      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       <���      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(��   max       @F\(�     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vMG�z�     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q            t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @��           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       <�j      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B �   max       B/�A      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B ��   max       B0;o      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�   max       C�D,      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?pIt   max       C��)      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          A      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�c      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?����+      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       <���      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�33333   max       @F\(�     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vMG�z�     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @�i           �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��S&�   max       ?����+        R<            	   3         @            
      	                  	                     "         7      7               (                     4      
   %                           
      NG/NN
�IM���NɊeO}=NG�@O���P�fJN�өN.U�N�F�O�_N6�/N�;�OݒN��MN�R�N��jN��O%�M��O�N��dN��>M�UOd�O�W�N��nN�	PO�.NÍ�O�G_N#��N���Nu۳N�=�P��N35�N���NVbtOU1�N� N@(P"\N
ULN^T�O��N6�NaO�N�5�NG'O6��N���Ok�TN�I(N�=HNQ&N�~Z<���<���;ě�;��
;�o��o���
�o�t��#�
��o���㼣�
��1��j���ͼ������������o�+�+�+�t��t���w�'',1�0 Ž49X�49X�8Q�8Q�@��@��H�9�L�ͽY��Y��]/�aG��ixս}�}󶽃o��+��+��7L��\)���������P���㽧�{��j#/:/-#}��������}}}}}}}}}}�� 
#'-'#
������(/<HUaelnsunaUH</'$(��������������������py���������������mhp�
0I{������{UI0#
�������

 ���������#)*56A?86,*MOZ[hkohe[UOMKMMMMMMrt�������������|wttr��������������������������������������������������������QUUX`ahnz~zna^UTQPPQ���������������������������	���������������������������GO[hstuy|~uth[YOLJIGenz���znbeeeeeeeeee|���������������yx||����������������������������������������"#%/<?<:/(# """"""""��������  ��������������#..*����������������������������������������������'/6HUanz�����zRH/#05BN[[^[YNB75-000000Uanz��������znaULJKU������������������������������������������������������������")69BKDB<60)=Ht���}zx{h[VOKEA;=Z[dht����thf[ZZZZZZZ��������	�������������������������������������������������������������������^gtx����ztgc^^^^^^^^7BHLMS[_g[NA5) 
����������������������������������������>N[gt�������tg[G>88>��������������������SUaijgaUOPSSSSSSSSSSqtx�����������vtljqq|��������}y{||||||||dgiqty���������tgbad<<DIUVWUTOIC<968<<<<���)/0+)"�������������������������003<ILUW]][YUID<;000��������������;<HKSUOH<446;;;;;;;;�t�h�l�j�t��t�t�t�t�t�t�t�t�t�t����������������������������������������Ňŀ�{�z�{ŇŔŜŖŔŇŇŇŇŇŇŇŇŇŇ���ݽѽнννнݽ����������������A�8�3�3�6�=�A�M�R�Z�f�y��y�u�n�f�Z�M�A�U�S�H�G�H�H�U�W�a�l�l�a�U�U�U�U�U�U�U�U����߿ؿѿ������Ŀѿݿ���	���"���������z�U�C�,�J�g������������������������D�D�D�D�D�D�D�D�D�D�EEEEEEED�D�D�"���	��������	���"�$�+�"�"�"�"�"�"�Y�M�Y�Y�_�e�r�}�~���~�v�r�e�Y�Y�Y�Y�Y�Y�������������Ŀѿܿ��������ݿѿĿ�������������������������������������������������	���"�%�+�"��	��������àÖÒÍËËÍÓàãèìïùúýùøìà�(�!�(�5�A�B�N�T�j�s�~���|�s�h�g�N�A�5�(��ƺƳƱƲƳ����������������������������������������������� � ����������$�/�/�'�$������������������������Ľнݽ���߽ݽҽнĽ����H�F�E�H�H�I�U�X�X�U�H�H�H�H�H�H�H�H�H�H�A�8�5�/�/�5�<�A�N�Z�`�e�g�i�g�f�Z�N�A�A��ݾ׾̾ʾžʾ׾�����������������A�A�<�9�A�N�Z�_�g�i�s�v�s�g�Z�N�A�A�A�A�<�;�<�F�C�H�L�U�X�U�O�H�<�<�<�<�<�<�<�<���������(�5�A�N�W�\�S�N�G�5�(��������ʼּ��!�.�4�:�?�?�:�.�����ʼ������
�����
����#�,�/�0�0�,�#����I�@�B�I�V�b�f�b�[�V�I�I�I�I�I�I�I�I�I�I�������z�E�5�+�(�2�H�Y�m���������������ŽĽ������ĽŽнؽݽ��ݽܽнĽĽĽĽĽĹ|�p�l�p�x�������Ϲ����
����ù����|�Ŀ¿��Ŀѿݿ޿���ݿѿĿĿĿĿĿĿĿ����������������������������������������׾ʾȾ����������ʾξ׾ھپؾ׾ʾʾʾʾʾ��a�\�T�L�I�H�@�H�T�V�a�h�m�q�v�m�a�a�a�a�����k�i�s�����������	�"�-�/�;�"��������ÓÓÓÔØàãìíîîìáàÓÓÓÓÓÓ�ܻٻлû��������ûûлٻܻ������ܻ��[�Q�O�E�O�[�h�t�u�t�q�h�[�[�[�[�[�[�[�[��������������������������%�&�%���/�-�"�"�"�/�;�H�K�H�;�:�/�/�/�/�/�/�/�/���������	�
�	�� ������������<�,�/�7�H�a�zÓßìù����������ì�z�U�<�Y�W�Y�e�i�r�|�~���~�r�e�Y�Y�Y�Y�Y�Y�Y�Y�ֺͺɺ������ƺɺҺֺ�����ֺֺֺֺֺ������������������������!���������0�,�#�!�"�#�0�2�;�5�0�0�0�0�0�0�0�0�0�0���
��"�.�9�9�.�"�����������#��#�)�0�8�:�<�I�U�U�Y�U�P�I�G�<�0�#�#�������������������������������������������������¿¸´¿�������������	������������������(�1�4�7�4�(������������޼ټ������!�-�7�:�C�:�!�������������������������������������彫���������������������ĽνɽĽ�������������~����������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� _ G Y F B \ \ @ @ � A P \ ^ r � V k H 9 v   V @ \ P c ] E [ / h f g O x r l D M Q ` ^ f L R ? x D d e Z \ T < 8 s )  w  <    �    U  9  �    �  �  L  p  �  �  �  �  �  �  e  X  4  �  �  ;  �  �  P  '  �  �  �  G  �  �  �  q  �  �  r  �  G  �  �  ;  [  �  x  c    X  �  �    �  �  o  �<��
<�j;D���D���@��t��+��t��t��D������`B��j���,1�o�+��P��P��w�t��0 Ž8Q���'�%�����%�<j�Ƨ�L�ͽ����<j�D���T���q���� ŽT����+�u��hs�ixսq����;d��O߽�hs�����\)�����hs���P�����E���
=��-��^5��Q��S�B�WB�B��B�BLXB �B �B&�gB�B/�AB<,B
��B!78B��Bp�By�B_�ByB�B��B�B ,BLB��B8mBC�B-�9B
YB�B3�B�+B�B*"B�B!!PBgAB�iB��B�B,qB�BB�B	�DBHB"��B#<zB�`B�B�B
t�B
8\B
�B&�<B B<B&�`B]�BLB2�B��B��B��B>�B ��B ��B&ɗB<�B0;oB@"B
�B!��BɟB��B?�BA�B�TB�&B��BB
��BC�B?�BA~B/B.?,B�B@B=�B�.B>WB)�@B�QB!?vB@yB?3BB1�B�tB�)BV�B	A�B��B"��B#@�B��B@�B��B	�hB
<B	��B&�OB=.BxMB&�BCbB��A��\A�ƩA��~A,LJA=��A��hA���A���C�D,A\F?쉉Azh�A.aCA���A�r*A��B?Q�B	YwA'�Aĥ�A�o^AT��A�7�A�Q:A��~A�tA���B�YA��A(��>�A{|]A��AP�)A��A��A�Jj@��5A�ŭA��ZA�GAX��A�Q�?���@;k?A��A�M	A_N�A�b A��A�$A4ȰA	�-B�A#>�A )�C���A��A�uA�usA-	�A=fAŉ�A��A���C�9�AZ��?���Ay�5A.DRA���A��>A��EB��?pItB	�+A'�uA��#A�&CAU_A��A�x�A�|5A
�YA�,B��A�bA(�,C��)A{�RA��AP�A�} A��_A�~�@��A�߮A���A�1qAW�A�K4?�N�@4��A�u�A�yA_@A��A�hfA��A4��A	 �B�[A"�)A!&�C��            	   4         A            
      
                  	                     #         8      8               (                     5      
   &                            
                           %   ;                                                         +         5      %               -                     )         #                                                         =                                                         '         #      %               +                     #                                          NG/NN
�IM���NɊeO4kaNG�@OA�P�cN�өN.U�N�F�N���N6�/N�;�OݒN��MNL�SN��jN��O%�M��O�N���N��>M�UOd�O��N��nN�	O�DRNÍ�O�G_N#��N���N.ĽN�=�O�N35�N���NVbtOU1�N� N@(O��^N
ULN^T�OA��N6�NaO�N�5�NG'O6��N���O&+PN�I(N�=HNQ&N�~Z  |  �  ,  	  	�  �  �  s  	�  t  �  �  �  �  �  �  �  �  a  �  �  .  �  �  �  b  l  �  �  6  9  �  T  ?  G  �  p  �    k  �  L  �  c  �  P  �  �  �  �  �  �  �  �  �  �  �  �<���<���;ě�;��
���
��o��C��t��t��#�
��o��9X���
��1��j���ͼ�/���������o�+�C��+�t��t��#�
�''u�0 Ž49X�49X�8Q�<j�@��P�`�H�9�L�ͽY��Y��]/�aG���O߽}�}󶽙����+��+��7L��\)���������
���㽧�{��j#/:/-#}��������}}}}}}}}}}�� 
#'-'#
������,/<HU`ahkiaUH<5/*(,,��������������������}����������������{}}
#0Ib{������{U<#��
�����

 ���������#)*56A?86,*MOZ[hkohe[UOMKMMMMMMz������������{xzzzz��������������������������������������������������������QUUX`ahnz~zna^UTQPPQ���������������������������	���������������������������GO[hstuy|~uth[YOLJIGenz���znbeeeeeeeeee|���������������yx||����������������������������������������"#%/<?<:/(# """"""""��������  ��������������",-*����������������������������������������������;<>IPUanz}���}oUHE<;05BN[[^[YNB75-000000Uanz��������znaULJKU������������������������������������������������������������")69BKDB<60)BMt��zwuxwsk[OIGB>=BZ[dht����thf[ZZZZZZZ��������	�������������������������������������������������������������������^gtx����ztgc^^^^^^^^)8>EIJLQOB5)#����������������������������������������?DNT[gr|���xtjg[NB??��������������������SUaijgaUOPSSSSSSSSSSqtx�����������vtljqq|��������}y{||||||||dgiqty���������tgbad<<DIUVWUTOIC<968<<<<���
&'%�������������������������003<ILUW]][YUID<;000��������������;<HKSUOH<446;;;;;;;;�t�h�l�j�t��t�t�t�t�t�t�t�t�t�t����������������������������������������Ňŀ�{�z�{ŇŔŜŖŔŇŇŇŇŇŇŇŇŇŇ���ݽѽнννнݽ����������������A�=�7�6�9�A�B�M�Z�f�o�r�p�j�f�a�Z�M�A�A�U�S�H�G�H�H�U�W�a�l�l�a�U�U�U�U�U�U�U�U������޿ݿۿݿ�����������������{�e�V�D�9�2�M�g����������������������D�D�D�D�D�D�D�D�D�D�EEEEEEED�D�D�"���	��������	���"�$�+�"�"�"�"�"�"�Y�M�Y�Y�_�e�r�}�~���~�v�r�e�Y�Y�Y�Y�Y�Y���������¿Ŀѿҿݿ����ݿѿĿ�����������������������������������������������������	���"�%�+�"��	��������àÖÒÍËËÍÓàãèìïùúýùøìà�(�!�(�5�A�B�N�T�j�s�~���|�s�h�g�N�A�5�(��ƾƳƲƳƴ����������������������������������������������� � ����������$�/�/�'�$������������������������Ľнݽ���߽ݽҽнĽ����H�F�E�H�H�I�U�X�X�U�H�H�H�H�H�H�H�H�H�H�A�8�5�/�/�5�<�A�N�Z�`�e�g�i�g�f�Z�N�A�A��޾׾;ʾƾʾ׾�����������������A�A�<�9�A�N�Z�_�g�i�s�v�s�g�Z�N�A�A�A�A�<�;�<�F�C�H�L�U�X�U�O�H�<�<�<�<�<�<�<�<���������(�5�A�N�W�\�S�N�G�5�(��������ʼ׼��!�.�2�:�>�?�:�4�.�����ʼ����
�����
����#�,�/�0�0�,�#����I�@�B�I�V�b�f�b�[�V�I�I�I�I�I�I�I�I�I�I�����z�T�H�A�7�7�D�H�m�z�����������������Ľ������ĽŽнؽݽ��ݽܽнĽĽĽĽĽĹ|�p�l�p�x�������Ϲ����
����ù����|�Ŀ¿��Ŀѿݿ޿���ݿѿĿĿĿĿĿĿĿ����������������������������������������׾��������Ⱦʾ˾׾پ׾׾ʾ����������������a�\�T�L�I�H�@�H�T�V�a�h�m�q�v�m�a�a�a�a�����o�q�������������	��&�!������������ÓÓÓÔØàãìíîîìáàÓÓÓÓÓÓ�ܻٻлû��������ûûлٻܻ������ܻ��[�Q�O�E�O�[�h�t�u�t�q�h�[�[�[�[�[�[�[�[��������������������������%�&�%���/�-�"�"�"�/�;�H�K�H�;�:�/�/�/�/�/�/�/�/���������	�
�	�� ������������H�<�5�5�<�H�aÇÓàìùþûôàÏ�n�U�H�Y�W�Y�e�i�r�|�~���~�r�e�Y�Y�Y�Y�Y�Y�Y�Y�ֺͺɺ������ƺɺҺֺ�����ֺֺֺֺֺ������������������������������������0�,�#�!�"�#�0�2�;�5�0�0�0�0�0�0�0�0�0�0���
��"�.�9�9�.�"�����������#��#�)�0�8�:�<�I�U�U�Y�U�P�I�G�<�0�#�#�������������������������������������������������¿¸´¿�������������	������������������(�1�4�7�4�(���������������������!�(�.�1�3�.�!��������������������������������������彫���������������������ĽνɽĽ�������������~����������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� _ G Y F 8 \ C > @ � A 9 \ ^ r � < k H 9 v   R @ \ P \ ] E 4 / h f g : x s l D M Q ` ^ j L R  x D d e Z \ N < 8 s )  w  <    �  �  U  M  �    �  �  �  p  �  �  �  v  �  �  e  X  4  �  �  ;  �  k  P  '  �  �  �  G  �  A  �  �  �  �  r  �  G  �    ;  [  �  x  c    X  �  �  q  �  �  o  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  |  �  �  �  �  �  �  �  �  �  �  �  m  X  @  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ,  )  %  "        �  �  �  �  _  8    �  �  �  _  3    	     �  �  �  �  �  �  �  �  �  |  o  b  U  F  6  '      	I  	�  	�  	�  	�  	�  	�  	�  	b  	+  �  �  "  �  ,  �  �  �  Z  �  �  �  �  �  �  �  �  �  �  {  _  C  (    �  �  �  �  R  $  �    +  >  P  ^  r  �  �  �  �  x  P  )  �  �    t  �  �  e  d  @    �  �  �  E  �  �    9  �  �  [    �  Y  �   �  	�  	�  	}  	e  	F  	"  �  �  z  &  �  M  �  J  �  %  �    �    t  j  `  V  L  C  9  /  %         �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  c  D  &  
  �  �  �  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  T  5    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  _  K  7  "  
  �  �  �  t    �  �  �  �  �  d  A    �  �  �  �  g  >    H  2    �  �  �  �  �  �  �  �  ~  u  k  b  M  6    
  �  �  �  �  �  t  O  �  �  �  �  �  �  �  �  }  j  W  C  "    �  �  Y  
   �   `  �  �  q  `  L  6  8  =    �  �  �  �  R    �  �  �  M  �  a  O  <  ,      �  �  �  �  �  w  `  I  2      �  �  �  �  �  �  {  s  h  V  D  1      �  �  �  �  �  �  �  {  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  .  *  &        	     �  �  �  �  �  �  �  �  �  h  F  #  �  �  �  �  �  �  �  �  g  H  &     �  �  �  R    �  C  �  �  �  �  �  �  �  �  �  {  k  [  K  8  $     �   �   �   �   [  �  �  �    B  a  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  F  0    �  �  �  �  �  �  _  5    �  �  S  �  �  @  �  k  V  %    
  �  �  �  ~  T    �  �  e  !  �  w    �    �  [  "  �  �  ,  1      �  �  �  [    �  x  '  �  z    �  �  �  �  �  �  �  �  {  e  O  9    �  �  �  �  |  [  9  �  �  �  �  !  )  1  ,    �  �  �  k  +  �  g  �  �  "  �  9  4  0  +  #          �  �  �  �  �  �  s  T  2     �  �  �  �  |  j  �  �  {  T  (  �  �  �    y  �  �  �  �  h  T  L  C  ;  2  *  !         �   �   �   �   �   �   �   �   �   �  ?  =  ;  9  7  5  3  /  *  &  !          �  �  �  �  �  ;  ?  D  G  F  F  9  "    �  �  �  �  �  |  c  J  /     �  �  �  �  �  �  �  �  �  �  s  W  <    �  �  8  �  �  J   �  >  i  p  i  T  4    �    �  �  i  '  �  �  �    �  �   �  �  r  Z  B  *    �  �  �  �  �  �  �  w  i  Z  L  =  /         �  �  �  �  �  �  j  E    �  �  �  �  [  :  �  8  �  k  w  �  �  �  �  �  �  �    6  f  x  �  �  �  �  3  n  �  �  �  �  �  �  �  �  �  �  g  H  %  �  �  �  V    �  `  �  L  @  4  (        �  �  �  �  �  �  �  z  _  E  +     �  �  �  ~  q  d  V  C  0    
  �  �  �  �  �  �  �  �  �  �  5  8  ?  F  a  V  .  �  �  �  �  _     �  t    9  S  `  �  �  �  �  �  �  �  �  �  u  i  ]  Q  D  8  +        �  �  P  A  2  "    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  Z  �  �  �  �  �  �  \  $  �  �  Q    �  e  �  %    �  �  �  �  u  g  V  D  3  "    �  �  �  j  A     �   �   �  �  �  {  n  a  T  F  7  (    	  �  �  �  �  %  V  �  �  8  �    v  n  f  ]  T  J  @  6  ,       	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  [  G  3      �  �  �  {  V  �  �  �  �  �  �  k  S  9    �  �  {  J  
  �  k     �  h  �  �  �  �  �  l  K  #  �  �  �  r  C    �  w    �  l    �  �  �  �  �  �  �  �  �  d  8    �  �  z  #  �  E  ]  Z  �  b  9    �  �  �  l  E    �  �  �  i  6    �  �  _  &  �  p  ^  K  8  %    �  �  �  �  �  �  v  `  H  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    
    �  �  �  _  $  �  �  k  .  �  �  �  V  3  8  ;  1  �  {  