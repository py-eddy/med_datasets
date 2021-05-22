CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�z�G�{     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��F   max       P��{     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =C�     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @E�z�G�     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vM��R     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @N@           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�(`         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��G�   max       <�9X     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�:a   max       B-�     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B-�     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ƙ�   max       C�Ez     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�w   max       C�V�     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��F   max       P�:b     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��1&�y   max       ?���o     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�/     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @E������     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vM�����     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @M�           �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���S���   max       ?���C�\�     p  a�               5            
                  	                        '   .         %      	            
               	      P   1                  $         :   
                     
      4               	            %               
   
OG�M�N�eyN�
�P2�hM���N֙�N8(,O*�N��N�1�N� �N�+�OrCN�X�N�oO܎�Nv�kO�2�N��N>;�N�q�PC�"P=�N��4O�5�P%adN8��N�]NV}vN�NN�ӐO/�OM�<N�9N�h�O��N�0�NŇbP��{PdN��@N�OrN�M�NS��N��BO�ήOhFN��P)aQO�gO�^8O5��O��Ne,eO`�yN
HN<�fN/5�P_�N�dM��FO$�DO#XN�|N��lN��uN2�WO��/N�s�OG�N�G�O��N��N�N=C�<ě�<�C�<e`B;o�o�D���ě��o�t��#�
�49X�D���T���u��o��o��o��C���C���C���t���t���t����㼣�
���
���
��1��1��9X��j���ͼ�������������/�����������o�+�+�C��\)�\)�t����#�
�,1�<j�<j�<j�D���P�`�Y��aG��ixսixսm�h�y�#��%��o��+��7L��7L��O߽�O߽��P���P���P���T����)16964,)�������������������������������������������!#$,/<=HJH?<4/)#!!!!������
	#'
��������������������������`almz|������zmaa````MNS[`gigg[NNMMMMMMMM�����������������������������������������������������������������������������!�������������������������������������������������!#%*+)%#
�3BP[kth[OJ:)���ghst�������tshggggggFKTam���������zm]THF!()++)%
	6BOV[][OB;6666666666TTY_ahmz�~ztqmga\TTT����������������������������������������ehpt���������thdaee����������������������
&%!/ASR</�����rt~���������tsrrrrrrdgt�����}tjg`]dddddd�����



����������������������������������������������������������������������emqz���������~zsmgbe��
 "
��������#/68:<><:/#Tamz�������zma]UTRRT�������������������������������������6BhuncB0'��������������������������������������������6BNR[cglg[NHB:666666bgmt������tggbbbbbbb��������������������in{�������{nefiiiiii����)-2)"������IOY[chuy|uth\[POEDI��������������������������	"!������cht��������������toc7<HUnuzvgeeaZUH?<55706CO_hjkeb^[OLHB;500�����
!#,#
������3<IU`YUJIG<833333333_ht����������th\[^]_TUXbejjb^UUSTTTTTTTT����������������������������������������������������������������������������NNO[]^[NNGNNNNNNNNNN0<ISUXXWVUQIH?<0*(+0��
#/2/+*$#
�������������������������EHUZafnpnmaUNJHCEEEE:<AHUaela`UH<:::::::xz}����������|zzxxxxXgt�����������tg\QRX����


����������NUgst��������tg[ONJN����������������������������������������!))66B64))5=?>85/)%��ֺֺϺκҺֺ޺����� �����������a�Y�U�U�U�a�k�n�n�r�n�e�a�a�a�a�a�a�a�a���z�������������������������������������M�D�A�4�1�4�4�=�A�H�M�P�Z�[�Z�V�M�M�M�M�Z�I�A�5�%�5�A�Z�s�������������������g�Z�`�Z�_�`�m�y�z�y�y�m�`�`�`�`�`�`�`�`�`�`�U�S�I�G�>�<�4�<�I�U�b�n�i�j�b�_�U�U�U�U�f�c�Z�S�Z�d�f�p�s�}�s�q�f�f�f�f�f�f�f�f��~�u�q�s������������ʾž�������������ѿͿĿ����ĿѿӿԿڿѿѿѿѿѿѿѿѿѿ����������������
���
�	���������������徱�������������������ʾо׾���׾ʾ��������������������������������������������T�M�H�B�A�C�H�T�a�l�m�z�}��������z�a�T�Ľ����ýĽнݽ߽����ݽнĽĽĽĽĽ��B�5�)�&����)�5�B�N�T�[�g�j�g�f�[�N�B�����m�k�m�{�������������ĿͿؿݿݿѿ��������������������Ŀƿ˿ſĿ��������������(������ӿͿҿݿ����5�%�$�!�(�5�6�(�f�f�s�v���������������������������s�f�a�X�\�a�k�n�s�z�u�n�a�a�a�a�a�a�a�a�a�a������ƳưƧƞƤƧƳƸ��������������������Šŏ�{�l�l�wœŠŭ�������*�B�E�%������r�qƁƘƳ������$�3�4�$�����ƴƧƎƁ�r���������}�x�x�x�������������������������׾̾����������׾�	��&�(�"��	����������ðìù������#�)�6�B�X�N�I�?�?�)����H�B�<�9�<�H�M�U�X�a�a�a�X�U�H�H�H�H�H�H���޾������	��	��������������������������������������������������ſm�h�`�]�\�`�m�y���������������y�m�m�m�mààßÝàìöù����ùìàààààààà�ۿѿĿ����Ŀȿѿݿ������������ā�x�t�n�j�tāčĚĦĳĽĸĳİĦġĚčā���������������
������
�������������#���#�/�<�A�H�N�U�_�a�U�H�<�/�#�#�#�#���������	�� �"�/�8�;�H�P�W�T�H�;�"����"� ������"�'�/�;�>�?�;�:�3�/�&�"�"����������!�*�.�.�.�+�$�!�������.��:�t�~�����Ľнڽ��2�=�����н��� ������������'�@�Y�a�h�n�k�Y�M�������|�����������������������������������������������������������������������������������%�)�"�����������
����&�)�0�)������������g�d�]�Z�Y�Z�g�s�~���������s�g�g�g�g�g�g�h�[�O�C�D�H�V�tčĚĦĚćĀĀăā�|�t�h�ù����������ùϹܹ�������������ܹϹ�������*�6�9�6�*������������������������ʼ����!�*�-�*�"����ּ��r�h�i�`�_�e�r�~���������������������~�r�;�3�'�#�"�/�;�H�a�m���������w�m�g�T�H�;���|��������������������������������������ɺĺ��ƺɺߺ����!�$�'�&�!�����-�%�"�(�-�:�>�F�H�F�?�:�-�-�-�-�-�-�-�-�������"�/�;�H�N�T�`�[�Z�T�H�/�"��������'�-�'��������������������'�*�'�$��������������������$�����������������������û����лܻ��4�@�Y�j�q�q�Y�@�����ܻÿѿпƿɿѿҿݿ������������߿ݿѿѿѿ�����&�*�6�8�6�6�*��������������~�����������Ľ˽нӽнĽ½���������D�D�D�D�D�D�D�D�D�D�EEEEEEEEED�����¿²±«²¿�����������������������˿����������������¿ĿɿȿĿ������������������������������ȾʾѾϾ̾ʾ������������������������ʾϾ׾���׾ʾ�����������ĦğĝĜĢĳĿ��������������������ĿĳĦ�$������$�0�;�=�F�I�L�I�=�0�$�$�$�$�<�/� �����#�/�<�H�J�P�R�V�Z�U�T�H�<�ɺǺǺɺֺ���������������ֺɺɺk�[�c�e�r�~�������������º����������~�k�����������������������������������������z�x�s�t�zÇÓàçìõìáàÓÇ�z�z�z�z 5 L O > V ( F . 6 n 0 w K 3 ! ` \ F V K Y c o g R K b <  d - M L : 3 D 7 @ S E 3 K j o 1 @ ` ) t E i > ^ M M  b \ Q y < J 9 j a G B i 8 C B Q p c '    S  /  �  �  �      A  r  g  �  .  �  �  �  9  U  �  �  /  �  �  L  �  �  �  =  ]  �  |  �  �  �  �  �  	  -    	  �  �  �  �  �  [  �  �  V  a  7  r  A  �  l  �  �  J  s  G  �      u  �  �  �  �  �  �  �  �  �  �  Z  <49X<�9X<D��<t��L�ͻ�o�49X�49X��t��49X�u��o��t���h�ě��ě��,1��t���P��h��/��j�ixս�o��h�0 Žixռ��ͼ�h��j�+���\)�0 Ž�P�0 Ž@���w�49X��G����
��P����P��P��P��\)�aG����\�L�ͽ�%�}󶽍O߽T���}�]/��%�u��/���T�u���罸Q콓t��������-��O߽����1������
�� Ž�Q����B��B/Bp,BnvB��B!�A�h�B£B!�B-�B įB[�Bm�BZ1B*
�Ba�BJB��B �PBs�B{�A�:aB1^BV�Ba B��B>�B�[B	�CB3_B*XB@�B)�:B $vB�BA�X�BI�B7�B2�B!;�B ��B@�B	�B�~B(�B��B��B�tB-%:B�B�B��B#��B&�iBf^B'�ZB 7�B
�B�]B�qBo�B&�B�KBUtB��B�<B�sB
n�B.jB	}3B�B�wB[�B�	B��B�B��B?+BܝB!h$A���B��B!��B-�B �B@�BH�B;�B)��B>�B�6B�7B �DB��BH�A���BL�BqBE�B��B��B�`B	��BK�B*BLBʍB)�A��EB�CB�A��BB�B�B��B!^B ��B&_B	�WB�rB)
�B��BA�B��B-6=B]BXHB��B#�
B&��B�gB'�QB >&B
�B��B��BDIB&QB?�BN�B�VBPB��B	�BA�B	��B�nB� B]�BA�@G;Aƿ�A��&A;�A��Aj��A�UeAAi�AI��Az�UA��AN6�AIHA�ǊA){A���Au|Aw�A��:AG܋A�qB�DA�*cBr@�]AU��A��%A���AW��A��AmP!A̞PAh�A��fA�ȿA�w�A��AA� |AA�A"�z@ϸ�A &+AsSA���A�]�A�)}A��>ƙ�A��A�e@��A��|A���@P��@u~�A���@�đ?���B��@���A}�A���A"XC�EzA��Au��AO�AP�6A�Z�B
G+A�D6@G-X@��A��A�97@ExAƏ�A���A<�VA�~AjރA�OAA��AK��AymA�	AP�AH�cA�yA)5A� sAv��Aw$�A��AIr�AƏBS�A�^�BCl@���AX�A�x{A�6�AXA[A��Al�jĄ�A��AދA���AĂjA�j�A��A<A�^@��A!�|Ar��A���A�t�A�߂A۾�>�wA�f�A��?�VA��XA�~@Tj@w;�A�i�@�;�?�۪B��@҉kA~ �A��A ��C�V�A���Au"EAN�AP7A�}mB
?lA�i4@PQ?���AсJAɄ6               5      	                        
   	                     (   /         &      	                           
      Q   2                  $         :                              4               	            &               
                  3                                    '      +            7   5      !   3                                       M   +                  !         /                              7                           !                                 %                                                      -         !   3                                       C                              -                              -                           !                  N�5M�N�eyN�
�O�aM���N֙�N8(,N��N��N�1�N� �Ntf�OrCN�X�N�J#O-.�Nv�kOS�ON��N>;�N�q�Po�O��N��UO���P%adN8��N��HNV}vN�NN�ӐO/�OM�<NM�N/�OP�N�0�NŇbP�:bO�*�N��@N�OrN�M�NS��N��BOs��OhFN��P$��O�gOVԟO"j�O��Ne,eO`�yN
HN<�fN/5�P
(N�dM��FO$�DO��N](�N��lN��uN2�WO��/N�s�OG�N�G�N��pN��N�N    �    !  �  �  �  q  t  0  �  �  �  L    �      !  �  �    Z  �  �  �  -  �  &  �  �  C  �  a  W  s    �    �  1  ?  p  �  6  b  �  �  �  �  q    1  �  �  Y  �  ~    �  ,  �  �    �  +  �  �  �  �  �  �    j  �<�/<ě�<�C�<e`B�D���o�D���ě��#�
�t��#�
�49X�T���T���u��C���/��o���ͼ�C���C���t���`B�\)���
��9X���
���
��9X��1��9X��j���ͼ������+�������,1�49X���o�+�+�C��'\)�t���w�#�
�8Q�@��D���<j�D���P�`�Y��aG��u�ixսm�h�y�#��+�����+��7L��7L��O߽�O߽��P���P���㽥�T���')0,) ����������������������������������������!#$,/<=HJH?<4/)#!!!!������ 	
����������������������������`almz|������zmaa````MNS[`gigg[NNMMMMMMMM�����������������������������������������������������������������������������
�������������������������������������������������	

#()($#
				")6BIFB:6)ghst�������tshggggggy��������������zonqy!()++)%
	6BOV[][OB;6666666666TTY_ahmz�~ztqmga\TTT����������������������������������������fhrt�����}thfbffffff����������������������
&%!/ASR</�����rt~���������tsrrrrrr_git~����ztmga______�����



����������������������������������������������������������������������emqz���������~zsmgbe��

������������#/055/# TWamz~�����zsmaZXUTT���������������������������������������6ee[6,!���������������������������������������������6BNR[cglg[NHB:666666bgmt������tggbbbbbbb��������������������in{�������{nefiiiiii����!&)������IOY[chuy|uth\[POEDI��������������������������	! ������cht��������������toc9<HUanqurnda]UE><77916BO]ijda][WOMIC;511����

���������3<IU`YUJIG<833333333_ht����������th\[^]_TUXbejjb^UUSTTTTTTTT��������������������������������������������������������������������������NNO[]^[NNGNNNNNNNNNN0<ISUXXWVUQIH?<0*(+0����
#,(&#
�������������������������EHUZafnpnmaUNJHCEEEE:<AHUaela`UH<:::::::xz}����������|zzxxxxXgt�����������tg\QRX����


����������NUgst��������tg[ONJN����������������������������������������!))66B64))5=?>85/)%�ֺԺҺֺغ��������������ֺֺֺֺֺ��a�Y�U�U�U�a�k�n�n�r�n�e�a�a�a�a�a�a�a�a���z�������������������������������������M�D�A�4�1�4�4�=�A�H�M�P�Z�[�Z�V�M�M�M�M�g�[�K�H�N�Z�g�����������������������s�g�`�Z�_�`�m�y�z�y�y�m�`�`�`�`�`�`�`�`�`�`�U�S�I�G�>�<�4�<�I�U�b�n�i�j�b�_�U�U�U�U�f�c�Z�S�Z�d�f�p�s�}�s�q�f�f�f�f�f�f�f�f��������{�u������������ž¾������������ѿͿĿ����ĿѿӿԿڿѿѿѿѿѿѿѿѿѿ����������������
���
�	���������������徱�������������������ʾо׾���׾ʾ��������������������������������������������T�M�H�B�A�C�H�T�a�l�m�z�}��������z�a�T�Ľ����ýĽнݽ߽����ݽнĽĽĽĽĽ��N�C�B�5�)�)��)�5�B�N�Q�[�e�`�[�N�N�N�N�������������������������������Ŀ̿ϿĿ������������������Ŀƿ˿ſĿ����������������ݿݿڿڿݿ����	�����������f�f�s�v���������������������������s�f�a�X�\�a�k�n�s�z�u�n�a�a�a�a�a�a�a�a�a�a������ƳưƧƞƤƧƳƸ������������������������ŭŜŒŅŐŠŭŵ�������(�+���������������������������$�)�+�)�$��������������������������������������������־ʾ����������׾�	��$�&�"��	����������ðìù������#�)�6�B�X�N�I�?�?�)����H�B�<�9�<�H�M�U�X�a�a�a�X�U�H�H�H�H�H�H����߾������	��	�������������������������������������������������ſm�h�`�]�\�`�m�y���������������y�m�m�m�mààßÝàìöù����ùìàààààààà�ۿѿĿ����Ŀȿѿݿ������������ā�x�t�n�j�tāčĚĦĳĽĸĳİĦġĚčā�����������
����
���������������������<�2�8�<�H�U�Z�Z�U�H�<�<�<�<�<�<�<�<�<�<��	������	��"�/�0�;�B�H�L�Q�H�;�/�"��"� ������"�'�/�;�>�?�;�:�3�/�&�"�"����������!�*�.�.�.�+�$�!�������u�G�;�:�C�|�����Žսݽ���)���н����������'�4�@�M�Y�f�h�h�a�Y�M�@�'������|�����������������������������������������������������������������������������������%�)�"�����������
����&�)�0�)������������g�d�]�Z�Y�Z�g�s�~���������s�g�g�g�g�g�g�[�Q�K�G�H�L�Z�[�h�tāčĂ�}�|��}�t�h�[�ù����������ùϹܹ�������������ܹϹ�������*�6�9�6�*�������������������������ʼ����!�)�,�*�"����ּ��r�h�i�`�_�e�r�~���������������������~�r�;�7�+�&�'�*�/�;�H�O�a�t�z�{�m�a�_�T�H�;���}�������������������������������������ɺƺúȺ������!�%�%� �������ֺɻ-�%�"�(�-�:�>�F�H�F�?�:�-�-�-�-�-�-�-�-�������"�/�;�H�N�T�`�[�Z�T�H�/�"��������'�-�'��������������������'�*�'�$��������������������$�����������������������Իлܻ���$�4�@�M�Y�h�o�q�o�Y�@�����Կѿпƿɿѿҿݿ������������߿ݿѿѿѿ�����&�*�6�8�6�6�*��������������~�����������Ľ˽нӽнĽ½���������D�D�D�D�D�D�D�D�D�D�D�D�EEEEEEED�����¿³²­²¿�����������������������˿����������������¿ĿɿȿĿ������������������������������ȾʾѾϾ̾ʾ������������������������ʾϾ׾���׾ʾ�����������ĦğĝĜĢĳĿ��������������������ĿĳĦ�$������$�0�;�=�F�I�L�I�=�0�$�$�$�$�<�/� �����#�/�<�H�J�P�R�V�Z�U�T�H�<�ɺǺǺɺֺ���������������ֺɺɺr�n�e�^�e�h�r�x�~�����������������~�r�r�����������������������������������������z�x�s�t�zÇÓàçìõìáàÓÇ�z�z�z�z ) L O > V ( F . B n 0 w O 3 ! ^ X F 8 K Y c p 3 > P b <  d - M L : 7 < , @ S E " K j o 1 @ P ) t E i + Z ? M  b \ Q r < J 9 h K G B i 8 C B Q d c '5  �  /  �  �  P      A    g  �  .  �  �  �  �  �  �  �  /  �  �  �  d  �  p  =  ]  �  |  �  �  �  �  p  A  �    	  �  8  �  �  �  [  �    V  a  %  r  �  �    �  �  J  s  G  e      u  W  �  �  �  �  �  �  �  �  �  Z    @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �       �  �  �  �  �  �  l  J  $  �  �     O  �  �  m  T  ;  "  
  �  �  �  �  �  �  �  �  �  �  �  z  s         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  !    �  �  �  �  �  �  �  �  �  {  e  M  4       �   �   �  �     Q  �  �  �  �  �  �  S    �  �  S    �  �    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  Z  M  A  4  &      �  �  �  �  �  �  q  k  f  `  Y  M  A  5  (    	  �  �  �  �  �  �  �  {  g  P  \  h  o  t  r  m  a  S  C  2      �  �  �  �  �  �  �  0  /  .  .  -  ,  ,  +  *  *  &           	     �   �   �  �  �  �  �  �  �  �    s  g  \  P  F  ;  1  '      �  �  �  �  �  �  �  �  �  �  �  �    z  o  a  R  C  9  /  &    �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  H    �  u  !  L  >  .    
  �  �  �  �  f  9    �  �  �  �  �  �  I   �    
     �  �  �  �  �  �  �  �  v  e  T  C  2  !    �  �  �  �  �  �  �  �  �  w  j  \  K  7  #    �  �  �  x  @    �  �  �  �  �  �  �        �  �  �  ~  ?  �  �  U  �  l                                          �  �  �                  
  �  �  �  �  d    �  p  �  �  �  �  ~  q  a  M  8  "  
  �  �  �  �  �  k  K    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h    �  �  �  �  �  �  �  �  l  T  <  "    �  �  �  �  r  R  K  S  R  L  V  Y  J  0  	  �  �  q  9  �  �    �  2  �  �    8  l  �  �  �  �  �  �  �  �  �  a  +  �  �    {  �  _  �  �  �  �  �  �  �  �  �  x  `  G  ,    �  �  �  �  �  �  �  �  �  �  z  [  6    �  �  �  �  �  �  �  �  a  0  �  �  -      �  �  �  �  O    �  �  �  i  #  �  �    �  ^  J  �  �  �  �  �  �  �  �  �  |  h  U  <       �  �  �  r  K     "  %  #           �  �  �  �  �  v  [  6    �  �  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  Q  7      �  �  �  X    C  >  8  1  "    �  �  �  �  �  s  W  ;  #    �  �  �  �  �  �  �  �  �  �  �  y  ]  >    �  �  �  �  �  r  a  \  X  a  T  D  3      �  �  �  �  �  �  �  h  L  ,    �  |  �      -  8  B  L  R  W  V  T  Q  H  <  *    �  �  �  �  Q  ;  8  8  6  5  >  N  `  w  �  �  �  �  �  �  �    �  *  �  �  �      �  �  �  �  �  �  �  l  M    �  �  M  �  �  7  �  �  �  �  �  �  �  �  �  �  �  �  |  m  ^  P  C  9  D  O    �  �  �  �  �  �  �  q  U  7    �  �  �     �  �  �  g  j  �  �  �  �  �  �  {  h  ]  K    �  �  �    �  �  �  r  �  �    #  ,  1  0  /  )  "    �  �  �  e  
  �  .  |  �  ?  4  )        �  �  �  �  �  �  �  �  u  b  O  <  (    p  f  ]  S  G  ;  .         �  �  �  �  �  w  X  	  �  m  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  o  `  Q  C  4  6  2  -  )  %  !                  �  �  �  �  �  �  b  b  a  a  a  a  `  `  _  ^  ]  \  Z  Y  V  S  P  M  J  G  4  ?  V  �  |  X  -  �  �  �  Y  '  �  �  �  `  �  "  �  �  �  �  �  �  �  �  {  j  W  @  *  !    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  |  �  �  �  �  �  �  �  l  D    �  m    �  I  �  J  �  �    q  h  _  X  R  N  L  N  J  :  1  0       �  �  f  '  :  V  �  �  �       �  �  �  �  �  �  �  b  >    �  �  �  v  s  -  1  -    	  �  �  �  �  �  �  n  U  ;  ,    �    �  f  �  �  �  �  �  �  n  R  1    �  �  y  C    �  �  �  _    �  �  �  �  �  �  �  {  g  S  @  .      �  �  �  �  �  �  Y  L  >  0  !      �  �  �  �  �  �  �  i  C    �  �  �  �  �  �  �  �  v  l  _  P  A  2  #       �   �   �   �   �   �  ~    �  |  x  p  i  a  `  i  S    �  �  v  ;  �  �  {  9          �  �  �  �  �  �  �  �  �  {  j  Z  T  Q  M  J  G  �  �  v  N  @  +  0    �  �  c  �  �  �  >  
  `  �  �  ,  (        �  �  �  �  y  N    �  �  Q    �  =  �  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Y  A  &    �  �  �  U    �  �  Q    �  >  �  N            �  �  �  X    �  �  �  C  �  �  ?  =  A  �  �  �  �  �          �  �  �  �  �  �  �  �  {  o  a  S  +      �  �  �  �  �  f  9    �  �  d  R  3  �  �  g    �  �  �  �  z  l  ]  K  7  "    �  �  �  �  R  2       ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       �  �  �  �  �  c  ;    �  �  �  \  (  �  �  '  �  �  �    �  �  �    i  S  >  )    �  �  �  �  K  �  �  N  �  �  3  �  �  �  �  �  v  ^  C  #     �  �  h     �  ]  �  t  �  �  �  �  �  �  �  u  f  Q  ;  $    �  �  �  �  �  }  X  3    �  �  �      �  �  �  �  u  H    �  �  {  3  �  �  Z    j  #  �  �  U    �  �  x  :  �  �  �  K    �  �  `  2    �  �  �  �  �  �  d  D  #    �  �  �  �  �    M  �    �