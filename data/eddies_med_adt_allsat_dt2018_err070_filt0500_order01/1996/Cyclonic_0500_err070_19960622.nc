CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�?|�hs     H  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P�.�     H  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��{   max       <�/     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F��
=p�     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=q    max       @vY\(�     �  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @O�           �  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @��          H  ;�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       <��
     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�     H  ?�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�_�   max       C��G     H  @�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�0   max       C���     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     H  D�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       PU�J     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:)�y��   max       ?̿�[W>�     H  H�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <�/     H  I�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F�          �  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vY�����     �  W�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O�           �  d�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�`         H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         BN   max         BN     H  f�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?̼j~��#     �  g�            6         )         	                                          S         6                  =   
         '   =            *         	               (      %      (            	   	               "         (         	            1         
         
      N�+*N�
�N��SP ��N���N�~O�(�NP��NE��O$��NH��N���O{p=O���OD�Nɒ�N�aZNO��N��XO&H*O�7O�N�aP�.�N�'N�%VO���N�|Nh�`N���O��wPBPPa�N[�N/�O�R�O���O��)N�?�N�i�O |O҈VO"��N��O��NN�;O��	NtãOA9`P
��NC%AO��PNښ�O�� Ok�XNf�N,��O4�wN�
�Nm��N�}`NW2vO��{ON	�N���O@�FO��_N�QMO9�PN�b�NB��O��@OJV�P$�N��TN_��NA>@P�O,N>(O"nTN���<�/<�9X<u<t�;�`B;�`B;��
;D��:�o$�  �ě��o�o�t��t��t��49X�49X�49X�T����t���t���t���t����㼣�
���
���
���
��1��1��1��j��j�ě����ͼ���������������������������������/��/��/��/��`B��`B��h��h���o�+�+�+�C��C��\)�\)�\)�\)�t��#�
�,1�0 Ž49X�49X�<j�<j�@��@��T���Y��m�h�q���q���q����7L��{��{���������������������')./)'����tz{�������zxwvtttttt��#0<IZfjlf`OI
�����������������������/6BOO[\`d_[WROCB61//UM</#
�����
/6>IOU;<HLUW\UHB=<;;;;;;;;#)/;<?</*#16=BILOSTWXg[OB63/.1��������������������()+)% ��
 #&/4540-$
��z�����������������zz������������������������������NT]adebcaTROKJNNNNNNWanz����znaXWWWWWWWW��������������������@BJ[grty{vtlg^[NBA@x{��������������zusxnz������������zxnkhn����������������������������)-$�������������������������nfbUPNNQUbdnonnnnnnn����)5>;7+������������������������?BOVZ[`[OCB;?????????HPTVY[ZXTHEB<<<????��������������������#0IU{�����{b<0
9an����������naZ:239KO[^hhih[OMKKKKKKKKK��������������������jx����������������ojamz���������ma]XVXZa���������  �����������������������������)36760)����������������������������� �������������������������������������������������|�����������������~|�������	

�������������������������������

���������������������������������������������������������������������#/<?CFEB<1#
R[gmtz�������tlg[SRR��������������������:B[homkeba][WOIB=98:R[ht|thh[WRRRRRRRRRRqt{��������tqqqqqqqqjmz~��������zxsligij������������������������������������������
#$'#!
���������������������������&05<IUnvyxobUI<0.1,&����������������������������������������)+*)*)����������������������������������������������� 
����������ztstt�������������$)56866-)&�����'��������������������������MU[gt��������{tb[OMM������������������������������������������������������������[g��������������[RS[����������������������������������������z�{zrnb`UHFCEHQU_anz
!#+/9/'#
	�
���	�
���#�/�1�6�:�2�/�#��
�
�
�
��ٺֺɺƺúɺ˺ֺ��������������Ľ½Ľнѽؽݽ������ݽнĽĽĽĽĽĽ����l�d�b�\�d�b�l�y�������Ľݽ�ֽĽ��������������ĿϿѿտԿѿĿ��������������������������ĿĿѿݿ���ݿۿѿпƿĿ��������� ������ŹŭŠŝŖŝŮŹ��������ĿĺĹĿ����������������ĿĿĿĿĿĿĿĿ�ݽܽнʽнսݽݽ������ݽݽݽݽݽݿ.�&�"��"�.�1�;�G�T�`�m�s�w�r�j�`�G�;�.���ݽнннֽݽ�����������������s�o�h�g�f�g�g�s�x���������������s�s�s�s�t�j�[�R�(������5�N�[�g�t�x�t��ѿſ������Ŀѿ���������������(���"�(�.�5�A�N�Z�c�e�g�g�_�Z�N�A�5�(����������������������������������������ā�{āąčĚĦĳļĸĳĦĚčāāāāāā�H�G�D�F�F�H�N�U�Y�[�Z�U�H�H�H�H�H�H�H�H�T�N�G�;�;�8�;�>�G�P�T�]�`�h�c�`�T�T�T�T�)��� ������������ �)�.�6�9�=�7�6�)������������
�#�0�I�U�I�>�0�#��
���������������������������������������������ż4�/�4�@�E�M�Y�\�Z�Y�M�@�4�4�4�4�4�4�4�4�������f�^�^�~����������!�,�.����ʼ��L�C�@�=�@�L�Y�c�e�i�r�t�r�e�e�Y�L�L�L�L�N�Z�b�_�Z�M�A�4�.�2�4�?�A�N�N�N�N�N�N�NùìÝÇ�|�q�o�q�s�zÇÓì������������ù��������������������	��	��������������U�P�P�U�a�d�n�y�y�n�m�a�U�U�U�U�U�U�U�U�N�C�N�S�Z�g�s�������������s�g�Z�N�N�N�N�����������ɾ׾������	�����׾ʾ��s�_�Y�W�P�F�Q�s�����������������������s���}�|���������ù���'�3�J�3�#��ܹù����Y�M�T�Y�e�e�r�t�}�u�r�e�Y�Y�Y�Y�Y�Y�Y�Y������(�-�2�(������������z�m�a�T�H�B�C�H�T�a�z�����������������zƬƜƚƟƧƳ�����������������������Ƭ�������������������$�0�5�8�8�0�+��ŠŜŕŕŞŠŭŰŹźŽ��ŹŭŠŠŠŠŠŠ�ܹعӹ׹ܹ���������������ܹܹܹܹܹܽн˽Ľ����Ľнݽ������� ����ݽноM�7�(��#�%�*�4�A�M�f�s������������s�M�������߿���������$� ����������������	�����������������������������������������������%�/�2�.�)�������������������������������������������޺ۺӺֺ������!�#�*�.�6�-�����⺽�������������ɺкѺκɺ���������������������������������������$��������ܺp�c�}���ɺֺ��$�)�!����ɺ������~�p���������������������������������������������r�g�a�_�b�g�s������������������������������������	��������	�������x�S�F�;�O�i�x�������ûӻ��ܻŻ����������|�{��������������� �����������������ù������ùŹϹϹѹϹùùùùùùùùù��s�p�g�Z�Z�Y�Z�g�s�v�x�y�s�s�s�s�s�s�s�s����������������"�)�/�2�2�/�"��	�������;�6�2�/�,�/�/�;�<�H�T�_�]�X�T�I�I�H�;�;�g�d�g�m�s�����������������s�g�g�g�g�g�g���
����(�)�6�6�6�0�,�/�)������Ϲʹ˹Ϲչܹ�����ܹϹϹϹϹϹϹϹϻлĻ��������������л��� �������ܻо����������������������ʾϾ׾پվ˾�������źŹųŹ�����������������������������ƽ��������������(�+�A�?�:�4�(�����������������ʼּ������ּ����������лͻû��������ûлܻ�������ܻлллм�������'�4�@�M�Y�c�c�_�Y�M�@�'��a�a�T�J�H�;�:�/�-�-�/�1�;�H�J�P�T�Y�a�a�ܻػл̻лѻܻ����������ܻܻܻܻܻܿG�?�;�.�"�)�;�>�G�W�y���������y�m�`�T�G�T�@�=�8�>�I�V�b�o�{Ǆǈǎǐǈǁ�{�o�b�T���z�y¦²��������#�����������������	���������������������ā�z�t�p�tāčĚģğĚčāāāāāāāā����������������������������������ĲħėēĕĦİĿ�����������
������ĿĲ�#��
�
��������������
���#�3�4�:�0�#������(�*�(�$����������������ĿſѿտݿݿڿӿѿĿ���������������E*E&E*E,E7E9ECEOEPEPE\EaE\E\EPEGECE7E*E* 2 2 _ " 3 v ] ^ j U ( 2 p " & 3 b ~ H X N L Q G : ) L 5 B j . 9 @ B K ? < . @ A 7 0 I B { c 1 ; 0 i s 7 S W v e ^ J Z \ p f 1 ( : 8 Y ; L I * > 3 K h L P ? S ( - O    �  �  �  �    �  w  �  �  ]  �  �  �  �  �  �  �  �  �  z  u  K  �  �  �  �  �  {  B  b  M  �  o  k  �  �  �  �  �  T  �  h  "  �    2  v  �  �  �      �  m  q  k  �  :  �  �  �  �  �  �  �  �  �  �    S  .  �  (  �  l  `  �  |  R  _  �<��
;��
<#�
�8Q�$�  ;o�\)�D�����
�t���C��T�������ě��e`B���㼛�㼋C���w�'\)��1����ě���/���P���o��`B�#�
�H�9��1�+��h�49X��o��-�o��w�ixս�C���w���t����,1�C���w��7L�o��+�'�t��u�'��0 Ž,1�'�w�#�
�m�h��\)�D���ixս���}�}�aG��q�������O߽��ͽ�o��%��O߽�1��7L�����S���"�B�zB��B��B&	�B1�B{&B��BF1B��Bn�B)�B��BjlB^LBx!BEA��CBӺB,�B݀B �B�iB)�iB,��B"��B'��B�5B||B�A���B�JB'	�B�B%�BaLBo�A�9�BN<Bl�B�gB!dkB"1�B*#�B,^lB��B��B"�B$S�B�B�Bo�B�jB	�TB�|B��B�cB�A��aB&B�B��B��B'>XB4�B0�Bm�B�oB"��B#BBG�BGBG�B��B	�B�B�B ٢B
�SB-B�]B5CBS>B��B��B�-B&?gB;�B�}B�5B@YB�/B6�B)��B�B�-BDIBEbB�A�}�B��B+��B�8B?RB��B)��B-0kB"j5B'�LB��Bp�B�A���B�/B&��B�BAcB�B��A�tNB@OB{�B��B!@@B"BlB)�%B,@
BC]B��B!B$IB��BH�B�BB	�vB=?B׾B��B=�A��B?B.�B_jB��B'2�B4�B?�B7jB�	B"�cB#=cB>IB?�BC�B�6B	�6B�qB��B �#B	�B��B��B=B?�A��@A�"A*�uA �:Ax�9Az�A�B\A�=A+�Ae��A,ŉA�vA��sA~�|A��aA��AߍmA�
�Aff�A�H�A逑A�:�@��A �]?֋�A<�A��A���Aƨ�A�
�AUkRA���=�_�?�}6A5��A�1�BnhB�A�6	?��A+ڪA?��A�_\A��3A���A���@Z�@/�A���@1�hA���A��VAZR�@�mA�?�>-�A��KA�r�A�_A�^kA�0>�{@��FAMe�A��A3Ȉ@�\s@���@���A� @��1Ai �Ba�A�2B�A�EQAӕ�A�O*A�|�A�Aw�gC��GA�u�@C3�A+�JA �MAx�"Ay�A���A㇤A,��Ac�?A,��A�e8A��A~�QA��iA�
aA�TuA�z�Ae@oAՀ�A�AυW@�KLAb�?��.A<�Aˀ�A���AƁ�A��eAT��A��Z=�0?���A4��A��B�zBÜA�B?�A+�A?vA���A�}�A�PA�A@_��@/�9A�|]@,P�A��A�:�AZ�@�l{A��y>=�A���A�tuA���A�t�A՚�>ݸA@���AM|A�m�A4�P@�<�@���@ѭ�A�w"@��Aj�BB�uA���B� A�AA�toA�q7A�(�A��Ax��C���            7         )         
                     	   	                  T         7                  >            '   >             +         
               (      &      )      	      
   	               #   	      )         
            1               	   
                  )         "                                                   =         !               +   5                           #         !               +            )                           #                                 +            )                                 "                                                   3                        +                                       !               '            %                           #                                 )            '            N�+*N\��N��SO��:N���N�~O�(�NP��NE��O$��N*�0N���OEPOa
�O�;Nɒ�N�aZN�<N��XO�;O�a�O�N�aPU�JN�'N�%VO�>N�|N+�lN���O�yPBPOiIN[�N/�O�R�O_�{OfqN�&SN�i�N�)�O�!gN�ܣN��O��NN���O���NtãO*��O��NC%AODT�N��)O�#HO �Nf�N,��O4�wN��#Nm��N�}`NW2vO�~�O,K�N` O@�FO�O�N��QO9�PN�b�N'��OR��O;�PT�N9�]N_��NA>@O�>oO,N>(O"nTN��      �  K  �    O    O  �  D  �     �    ~  '  �    �  �  �  d  =  �  �  p  �  i  L  y  v  �  �  �    '  �  �  b  �  _  5  �  b  �  \  6     �  �  <  A  �  �  �  6  �  �    �  �  H  g  0  ]  �  �      %  /  �  P  �  %  �    �  2    �<�/<�o<u�49X;�`B;�`B;��
;D��:�o$�  ��`B�o�49X��C��49X�t��49X�T���49X�u���㼛�㼓t��o���㼣�
�o���
��9X��1��`B��1�D����j�ě����ͽt��<j��/�������+��`B������/��`B��`B��/��h����h�\)�o�\)��w�+�+�C��t��\)�\)�\)�t��#�
�',1�D���8Q�49X�<j�@��P�`�D���e`B�ixսm�h�q���}�q����7L��{�� ���������������������))+)tz{�������zxwvtttttt#0<U\_^[VI<0#��������������������/6BOO[\`d_[WROCB61//UM</#
�����
/6>IOU;<HLUW\UHB=<;;;;;;;;#)/;<?</*#16=BILOSTWXg[OB63/.1��������������������()+)% 
#/02232-+#
�
��������������������������������������������������NT]adebcaTROKJNNNNNN[abnz���zna[[[[[[[[��������������������ADLNX[gt|zutjg][NDBAtz���������������zvtjnqz������������znlj�������������������������� '���������������������������nfbUPNNQUbdnonnnnnnn����)//,&������������������������ABOOX[^[OFB<AAAAAAAA?HPTVY[ZXTHEB<<<????��������������������#0IU{�����{b<0
:CNUanuwvnlhaUPHD<8:KO[^hhih[OMKKKKKKKKK��������������������jx����������������oj^acmz~�������zma\Z[^����������������������������������������)36760)��������������������������������������������������������������������������������|�����������������~|�����

���������������������������������

���������������������������������������������������������������������!#/<@BCB</#U[gqtv�������tpg[YUU��������������������@BO[hhieca_^[OMFA==@R[ht|thh[WRRRRRRRRRRqt{��������tqqqqqqqqjmz~��������zxsligij������������������������������������������
#$'#!
���������������������������)06IUbntxvnbUI<1021)����������������������������������������)+*)*)������������������������������������������������� 
����������ztstt�������������&)06766,)(���!#���������������������������NX[gt���������tg[QNN������������������������������������������������������������V]g�������������t]TV����������������������������������������z�{zrnb`UHFCEHQU_anz	
 #*/&#
					�
���	�
���#�/�1�6�:�2�/�#��
�
�
�
�ֺϺʺֺٺ�������ֺֺֺֺֺֺֺֽĽ½Ľнѽؽݽ������ݽнĽĽĽĽĽĽ����x�p�t�x�����������Žʽɽƽ��������������������ĿϿѿտԿѿĿ��������������������������ĿĿѿݿ���ݿۿѿпƿĿ��������� ������ŹŭŠŝŖŝŮŹ��������ĿĺĹĿ����������������ĿĿĿĿĿĿĿĿ�ݽܽнʽнսݽݽ������ݽݽݽݽݽݿ.�&�"��"�.�1�;�G�T�`�m�s�w�r�j�`�G�;�.�ݽѽؽݽ����������ݽݽݽݽݽݽݽݽݽ��s�o�h�g�f�g�g�s�x���������������s�s�s�s�X�B�0�)��%�)�5�B�N�[�g�t�~�t�g�X�ܿѿɿſĿȿѿݿ����������������5�)�(�#�&�(�3�5�A�N�Z�]�b�c�Z�Z�N�A�5�5����������������������������������������ā�{āąčĚĦĳļĸĳĦĚčāāāāāā�U�I�H�F�G�H�H�I�U�W�Z�X�U�U�U�U�U�U�U�U�T�N�G�;�;�8�;�>�G�P�T�]�`�h�c�`�T�T�T�T�)��������������)�-�6�8�;�6�-�)������������
�#�0�<�I�Q�I�<�0�#��	�������������������������������������������Ҽ4�/�4�@�E�M�Y�\�Z�Y�M�@�4�4�4�4�4�4�4�4����v�x�������ʼ���$�&���
����ʼ��L�C�@�=�@�L�Y�c�e�i�r�t�r�e�e�Y�L�L�L�L�N�Z�b�_�Z�M�A�4�.�2�4�?�A�N�N�N�N�N�N�NìèàÓÇÄ�{�z�~ÇÓàìü��������ùì��������������������	��	��������������U�R�T�U�a�i�n�s�v�n�k�a�U�U�U�U�U�U�U�U�N�C�N�S�Z�g�s�������������s�g�Z�N�N�N�N���׾ʾʾľþʾ׾��������	�������s�_�Y�W�P�F�Q�s�����������������������s�������������������ùϹܹ����ܹйù��Y�M�T�Y�e�e�r�t�}�u�r�e�Y�Y�Y�Y�Y�Y�Y�Y������(�-�2�(������������z�m�a�T�H�B�C�H�T�a�z�����������������z��ƶƳƦƥƧƬƳ��������������������������������������������$�'�/�1�.�&�$��ŠşŗŗŠŠŪŭŸŹŻŻŹŭŠŠŠŠŠŠ�ܹعӹ׹ܹ���������������ܹܹܹܹܹܽннĽ����Ľнݽ������������ݽноG�0�,�+�0�4�A�M�Z�f�s�y�����������s�f�G������������������� �����������������	�����������������������������������������������%�/�2�.�)��������������������������������������������ٺԺֺ������!�)�,�5�-�!�����⺽�������������ɺкѺκɺ������������������������������������!���������޺~�x�r�t�����ɺֺ��������ẽ�����~�������������������������������������������x�s�g�e�b�g�s���������������������������������������	�������	�	�����S�G�J�T�l�������������ܻû������x�_�S�����������������������������������������ù������ùŹϹϹѹϹùùùùùùùùù��s�p�g�Z�Z�Y�Z�g�s�v�x�y�s�s�s�s�s�s�s�s����������������"�)�/�2�2�/�"��	�������;�:�1�5�;�@�H�T�\�Z�T�S�H�>�;�;�;�;�;�;�g�d�g�m�s�����������������s�g�g�g�g�g�g���
����(�)�6�6�6�0�,�/�)������Ϲʹ˹Ϲչܹ�����ܹϹϹϹϹϹϹϹϻлȻ��������������л�����������о����������������������ʾԾ־Ҿʾƾ�������ŽŹŴŹ�����������������������������ƽ��������������(�+�A�?�:�4�(�������������������ʼּ޼����޼ּ��������ܻллû��������ûлܻ�����ܻܻܻܼ�������'�4�@�M�Y�c�c�_�Y�M�@�'��a�a�T�J�H�;�:�/�-�-�/�1�;�H�J�P�T�Y�a�a�ܻڻлλлֻܻ���������߻ܻܻܻܻܻܿG�=�;�7�;�H�_�m�y����������y�t�m�`�T�G�b�V�C�=�=�A�I�V�b�o�{ǃǈǍǏǈ��{�o�b���~�}¦²������������������������
�������������ā�z�t�p�tāčĚģğĚčāāāāāāāā����������������������������������ĿĳĩęĖġĦĺĿ�����������	������Ŀ�#��
�
��������������
���#�3�4�:�0�#������(�*�(�$����������������ĿſѿտݿݿڿӿѿĿ���������������E*E(E*E-E7E:ECEPE\E`E\E[EPEFECE7E*E*E*E* 2 ( _ # 3 v ] ^ j U 3 2 _   3 b ~ H T L L Q < : ) C 5 M j < 9 7 B K ? 9  B A 4 1 E B { N . ; / c s / P V _ e ^ J S \ p f 2 & ) 8 V : L I , . / H u L P < S ( - S    f  �  F  �    �  w  �  �  3  �  �  �  J  �  �  �  �  M  Y  k  K  �  �  �    �  X  B  J  M  �  o  k  �  �  �  �  �    b  "  "  �  �    v  w  �  �  �  �  H  �  q  k  �  �  �  �  �  �  p  g  �  W  �  �    ?  �  �  �  �  l  `  4  |  R  _  �  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN  BN    
    �  �  �  �  �  �  �  �  �  �  w  V    �  �  Q    �  �  �                �  �  �  �  ^  (  �  �  ]  
  �  �  �  �  �  �  �  �  �  �  �    u  h  [  N  >  ,    
  �    o  �    5  F  K  F  5    �  �  �  G  �  i  �  �   �  �  �  �  �  �  �  �  o  ]  F  .    �  �  �  w  I     �   �         �  �  �  �  �  �  �  �  �  �  �  �  �  {  s  k  d  O  1  /  "  	  �  �  �  ]  '  �  �  \    �  U  �  �    �    �  �  �  �  �  �  g  G  &  �  �    �  i    �  y  /   �  O  H  B  ;  0  #    1  ^  �  �  �  �  r  c  T  E  5  $    �  s  [  E  0    
             �  �  �  �  �  �  �  m  8  ?  E  I  N  R  F  7  %    �  �  �  �  �    \  3  	  �  �  �  �  �  �  �  �  �  �  �  z  p  b  Q  @  /     �   �   �  �  �        �  �  �  �  [  .    �  �  �  �  x  8  �  p  C  i  |  �  �  �  �  �  �  �  z  ^  6    �  �  Q  �      �  
              �  �  �  �  �  �  �  Y    �  >   �  ~  |  y  w  t  q  m  j  g  c  `  \  X  R  M  G  @  8  0  (  '      �  �  �  �  �  �  �  q  [  F  3  !    �  �      E  Y  m  x  |    y  t  m  g  [  J  8  !  	  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  L    �  �  C  �  i  �  |  �  ,  �  �  �  �  �  �  �  �  S    �  �  P  F  2    �  	  F   o  �  �  �  �  �  �  �  �  �  �  �  z  `  E  )    �  �  _  -  d  c  b  b  a  `  `  \  U  O  I  C  =  5  *      	   �   �  �    +  <  9  *    �  �  c    �  O  �  x  �  I  |  z   �  �  �  �  �  }  o  `  Q  B  4  &    	  �  �  �  �    4  T  �  �  �  �  �  �  �  ~  j  S  <  #    �  �  �  w  J     �  �    M  f  p  o  m  i  c  K  %  �  �  >  �  A  �  �     �  �  �  �  �  �  �  �  �  �  �  �  z  ]  7    �  �  l  /  �    7  Q  f  p  t  o  e  X  K  >  0  !      �  �  �  �    L  D  <  4  (        �  �  �  �  �  �  �  q  ]  ;    �    +  6  E  W  c  r  y  p  `  H  *  	  �  �  �  �  V     /  v  h  S  9    �  �  �  `  $  �  �  �  x  T  ,  �  �     �  �  �  �  �  N  �    �  �  �  �  \  �  �  �  M  �  �     f  �  �  i  O  7  &    �  �  �  �  s  K  #  �  �  �  }  Q  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      	      �  �  �  �  �  �  �  c  B    #      �  �  �     �  O    "  &  '  &  '      �  �  �  k  &  �  ^  �  i  �    J  $  |  �  �  �  �  �  �  �  �  �  Q    �  1  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  j  b  b  \  U  L  @  0      �  �  �  �  �  t  V  3    �  �  �  �  �  �  �  �  �  �  k  M  %  �  �  t  !  �  J  �    I  �  %  H  Z  _  \  T  L  =  *    �  �  �  �  F  �  S  �  %   �     *  1  5  /  '        �  �  �  �  �  �  i  0  �  �  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  |  x  p  h  _  W  N  b  ^  Y  Y  [  V  O  P  T  M  >  .      �  �  �  o  _  O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  @    \  \  V  K  <  *      �  �  �  �  �  �  h  H  #    �  �  6  *      
     �  �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  �  u  ^  E  %    �  �  �  H    }  �  �  {  k  Q  -    �  �  u  5  �  �  �  i  ?    �    �  �  z  r  j  b  [  O  A  3  %    
              	  �    .  ;  4  '    �  �  �  �  U    �  t    �  Q  �  �  >  ?  @  9  1  #    	  �  �  �  �  �  �  {  ^  ;    �  �  �  �  �  �  �  �  �  ^  A  "    �  �  v  )  �  I  �  U  E  �  �  �  �  �  �  �  �  }  ^  E    �  �  �  R    �  �  d  �  �  �  �  �  �  m  T  8    �  �  �  �  �  d  A    �  g  6  $       �  �  �  �  �  k  O  3         �  �  �  �  u  �  �  �  �  �  z  g  T  ?  )    �  �  �  �  �  _  .   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  m  R  8    �  �  �  �  �  �  �  �  t  _  C  '    �  �  |  N      �  �  �  �  �  p  ^  H  2      �  �  �  �  �  �  �  v  b  N  �  �  �  �  �  �  �    z  q  i  a  V  J  >  2  �  �  v  4  G  G  B  ;  2  (        �  �  �  �  }  X  :    �  �    [  `  b  e  a  Z  S  A  /    �  �  �  Y    �  ^  �  �   �  .  /  /  (      �  �  �  �  �  m  F    �  �  �  �  �  �  ]  V  E  4  "  �  �  �  �  Z  *      �  �  �  o  @  �   �  ~  �  �  �  �  d  L  ,    �  �  `     �  h  �  x  �  �  �  �  �  �  �  �  �  �  r  Y  G  9  .  '        �  �  �  �        �  �  �  �  �  �  �  n  J    �  �  F  �  �  �  F    �  �  �  �  �  �  �  e  J  /    �  �  �  v  J     �   �  #  %  %  #    �  �  �  �  �  _  >    �  �  �  �  ]  8      #    /  &          �  �  �  {  C    �  &  �  �  *  �  �  �  �  �  p  L  $  �  �  �  \  !  �  �  B  �  i  �    ,  H  P  F  5    �  �  �  �  U    �  �  !  �  <  �  �  �  �  `  E  :  T  �  �  �    6  �  !    �  �  �  �  t  V  7  %        �  �  �  �  �  �  �  y  `  F  +    �  �  �  �  �  c  @  !      �    �  �  �  �  �    _  =    �  �  �  �  �        �  �  �  �  ]  !  �  �  6  �  �  �  ,  �  �  �  �  �  �  �  �  s  d  ]  V  K  <  ,      �  �  �  �  r  2      �  �  �  �  �  �  �  r  ]  J  4    �  �  �  �  �    �  �  �  n  W  :    �  �  �  v  9  �  �    �  �  1  m  <  �  �  �  �  �  �  f  >  	  �  �  C  �  �  d    �    �