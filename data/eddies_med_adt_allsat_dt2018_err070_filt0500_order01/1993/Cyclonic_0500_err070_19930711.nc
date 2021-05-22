CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�-V�     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��-   max       P��a     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       <�j     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @F�\(�     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vyp��
>     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P@           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�8        max       @�x`         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <o     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4��     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <��p   max       C���     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =���   max       C���     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��-   max       PV�     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�b��}W   max       ?�����t     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       <�9X     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @F���R     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vyp��
>     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P@           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�8        max       @�0@         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >   max         >     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?�����t     �  b�         	            �         
                                          	                                                         
      	                                    
      )   H               	      ,            1         
      N�3N�|MNC`O2[M���O�N�P��aOEA�O��N��M��vO%��M�sBO���N�͊O<ɔN�jBO|��Op�OQi�N�:M���P�qO�Nޑ9Oq��Or�ZP&O���N3YtN:&N��"O�t(N[e�N<�iM��-O;V�N��_N5N�AOy�NU8�N�JON�)�M�G\O��N?{�N��N��O	,mNU�ObW�N(iO>Nve�O:U@N�>WOYT�O�dQPV�N M�O	��O+bNR�O"rzN
��P��N��NNH	�OW�xO�`�O
�7N�:`N��HN��=Nx<�j<e`B<t�;��
;o�o���
�t��t��#�
�49X�49X�49X�D���D���D���T���e`B�e`B�u�u�u��o��o��C���C���C���t���t���t���t���t����
��1��1��9X�ě����ͼ�����/��/��/�o�o�o�C��C��\)�\)�t���P��P�#�
�'',1�,1�,1�,1�0 Ž0 Ž49X�@��H�9�L�ͽT���T���ixս�+��7L��\)��t����
��1��^5�ě�	!),))"						)-+)$""#'/;:80/."""""""""�����

����

<HUantroda]U<,$'����� " ���������|��������������{ywy|��������������������chpt��������tmhacccc������������������������������������������



���������������������������|{��GHU_adfhhda`UMH@ACGG_anxxz�����znga\YXZ_��������� ��������������������������_hht��������thf_`__��������������������16BNOVUOB:6.11111111fhtttxxthfffffffffff��������������������������������cnt����������~��{thc$6COZXSE@861*ku��������������kijk5NgtqjjaXNB) �
/DHLG<#
���� #,/6<><;/#"        _agmnz~|zna________nnuz��������zqnnnnn)6BKPRUSOB62%#)*5BBB>=5))########����������������������������������������mz�����������ztmhcdm������������������������������������������

�������S[ht��������tqa[UPPS��������������������CN[cgrt�����trg[NGCCYajnz��zzna\YYYYYYYY��������������������OVcmz����������mdWPO��������������������~�����������������~~��������������������[[egt���|xwvtgg[ZUT[ggmt����tggggggggggg��������������������[bnotzncbZ[[[[[[[[[[��������������������eht�������}theeeeeee����������������������������������������#0<BGJKLLIG<0.6<CIUXaegf`WI<70.-.u����������������qpu����������������������������������IOR[hptzwtpoh[UOMGII�����������������������������������������������������������������������������������������������8<DHU`\UHC<888888888)5BN[_\N5)��
#8<HTNFB</#
�����������������������FHUahjeaUPJHFFFFFFFF���
������������������������������������������������źɺȺƺɺҺֺ�������ֺɺɺɺɺɺ�������������������������	� �������������������������������������������������������ؾʾ����������žʾ׾��������𼘼��������������������������������������;�,�*�)�&��"�;�H�a�u�z�}�~�z�m�a�T�H�;���ۻԻֻ�����f�����������r�M�4�'����������������(�5�8�=�?�9�5�(���������ݿѿĿ��������������Ŀٿݿ�������<�8�/�,�$�/�9�<�H�L�U�Z�Y�U�M�H�<�<�<�<ĿĺĽĿ������������ĿĿĿĿĿĿĿĿĿĿ�.�(���(�/�4�A�M�Z�f�s�v�u�~�s�Z�M�A�.���ݽܽݽ��������������������B�6�0�)���)�B�O�[�\�d�t�~�{�t�h�[�O�B�A�@�8�A�E�N�Z�g�s�������y�s�o�g�Z�N�A�A��������������������������������������������������������������������������������àÓÏÊÇÁ�zÇÓàáìù������üùìà���y�x�s�n�t�x�����������������������������������������*�6�C�O�U�R�D�6�*�����U�J�K�U�U�a�n�s�r�n�i�a�U�U�U�U�U�U�U�U�׾Ծʾʾʾ׾����׾׾׾׾׾׾׾׾׾��b�I�#�
������0�U�n�{ŇŔŠŬůŪŇ�n�bŇń�{�w�w�z�{ŇŔŠŭŴŲŭūŨŠŔŌŇ������²²·¿�������������������������ؿ.�"�	����پ׾�����"�.�3�;�B�G�M�G�.���������������������ʾ׾ܾ����׾ʾ�����������������*�C�Q�S�I�C�6�*������������%�)�.�6�B�W�^�h�h�]�O�B�6�)���H�B�<�3�<�A�H�R�U�\�V�U�H�H�H�H�H�H�H�H����� �������$������������������������������������������������޾s�f�Z�B�M�V�b�s���������������������s�����������������������������������������G�F�;�G�T�`�h�a�`�T�G�G�G�G�G�G�G�G�G�G����������������������������������������ƽƺ��ƿ����������������������������ƽ����۾ݾ�����	���	����������������������"�������������������������$�0�;�4�0�$�"�������������|�}�}������������������������������������������������������������������������������������	��������	���������������������������������������FF FFFF$F%F'F$FFFFFFFFFFF���������������������	��"�(�*�%�"��	���;�1�;�@�H�T�a�f�a�^�T�H�;�;�;�;�;�;�;�;�����������������������������������������O�I�H�M�K�O�[�[�h�m�t�x�t�h�\�[�O�O�O�OĦĦĚĘĐĔĚĦĳĿ��������������ĿĳĦ���������������������������������)�5�G�Q�N�E�8�)�"���N�I�L�N�Z�g�p�g�f�Z�N�N�N�N�N�N�N�N�N�N�
����ۿҿؿݿ������ �(�*�+�(��
������'�3�3�4�4�3�2�*�'�������V�O�I�F�C�=�2�=�I�V�b�o�ǉǈ�{�w�o�b�V�$���� ��	����$�.�0�*�$�$�$�$�$�$�����������������нݽ�����ڽĽ��������A�����������4�A�M�Z�e�h�f�a�Z�M�A�����r�L�E�L�Y�������ֺ������ֺ������S�I�F�:�-�$�-�:�F�S�_�`�_�X�S�S�S�S�S�S��	������������"�.�3�;�B�;�.�"��ù������������ùϹӹܹ�����߹ܹϹù����
�
�	�
��#�$�,�#�#�������������ݿӿѿĿ��������Ŀѿݿ�������'��'�4�7�@�M�R�M�C�@�4�'�'�'�'�'�'�'�'�Y�Y�r�������ֽ������ ��ּ����f�Y�ֺպɺ����ɺɺֺغ�������׺ֺֺֺ�E�E�E�E~EzE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������������������������������������EEEEEEEEEEE7ECEPE_EdE^EUECE*E�a�g�a�`�T�H�;�/�/�&�#�/�;�B�H�M�T�a�a�a�������������ùǹ˹ù���������������������������!�$�.�9�.�*�!����������������(�5�A�B�A�<�5�(�%�������������������������������������������� 0 I R V 5 ; 1  u * b U h H A L C ] . d B P Z 6 H x R 6 I ' U & S B < c : b P c " 3 m Q Z 4 _ � H I ^ N S g R / 3 K = O y z * ; t m v 1 o � B   3 2 < ?  �  �  h  �    �  ;  �  �  �  5  �   �  3  "  �  �    X  
  �  
    b    ^    �  J  G  p  �  @  n  I     �  �  L  (  �  j  G  �  	  �  y    �  J  [    R  �  �  �  �    ,  �  �  �  (  p  �  K  �  �  }  �  �  %  �  �     =;�`B<o:�o��o�D���T������/���
����T����1�u�#�
���ͼ��ͼ����w�\)��`B����C��0 ż��ͼ�9X�����8Q�49X��j���
�ě��C����ͼ����ě���h��h����P�T�����'�w�#�
�ixս#�
�',1�H�9��w�P�`�,1�T���T���ixսT����7L����@��aG���O߽T���m�h�aG��\��o��hs���w��F���`��-�����l���"�B��BA���B#�WB$}�BN�B^�B
��B'Bu6B�B/@B��B<B��BūBsKB;�B�IBS"BH�BB�\B�B��B/�>B4�B��B��B5B�fB��B��B2"B&&B��B �B#�B1�B�B8B!�B	AB�Bi�B oB��BvBv�B	o�B	�B�]B'��B./BFgB�(B��B%�sB&��BZ�B�B�5Bb�B�B*i�B)��B,��B!�@BR�B5YB= B�B��B��B�$B��B��B�A��B$>-B$��BC�BDB
ƹB>}B�$B ��BB��B@B�BìB��B�\BQ�B@0B@�B@MBvB�B(�B0:�B4��B��B��B>�B�+Bh�BśB@�B=&BȩA�RB�B?lB�gB��B!�IB	��BBBD=B �iB=AB
��B_�B	@B	�B@7B(SB�$B@"B��B�B%��B&�	B>oBJB�HBA�BB)�WB)�MB-:B!E*B��B��B@B�oBϚB��B��B��@?8�A�IA���ARv�@�g�A��@�~�A���Ax��A���A㌎A=�A-�A�zA���A���A�I"A�r�@���A�M�AƐwAS��A�1�A�eA���A[ǈAPUA�L�A׳�A��AԖ�A��AD��A�j�Af�Apt/B�yAX24A�uB	A�d�A#�AZ0�A��lC���A��A��A��A�|�A��A��A��A���A�H$?��BKB	05A$�A9�@"�@���A\H->��_A��A~m@�p�@���@=q�C�#AѻIC���A�GB<��pA,pA��eA�Za@C�A��A�IASS@���A��-@� hA��~Ax��A� �A⫥A<�A- A؀@A���A��$A�}EA�Aa@���A��HA��AS/A��A�A�o_AZ��AO�A�6A�]rA�@kA��AОAF��A���AgJ�Ap�B��AXB8A���B�A��	A#߆A[�A���C���A��OA�\.A��{A�wFA�A�3�A�s�A��uA~�?��B��B	A�A"�A9yj@�@�ſA\�$>��A�ZA�<@�h:A Ѡ@;��C���A�~C��$A���=���A�|A�r�A��
         	            �         
                                          
                                                         
      	                                          )   I               	      ,            2                                    5                                                /               %                                                                                                5                     3                                                                                                               !                                                                                                5                     /                           Nk�]N�|MNC`O2[M���O�N�O�N�]�N���N�[�M��vO%��M�sBO8:2N� ,O<ɔN�jBOC�'O�3O6c�N|�M���O�@�O�\NHG�Oq��OC��O�R�O\�WN
f�N:&N��"O�t(N[e�N<�iM��-O;V�N��_N5N�AOy�NU8�Nʥ�N�)�M�G\O���NªNIx�N��N��NU�O(�N(iO>NF��O(ejN^MXO#֘OM��PV�N M�O	��Np1�NR�O"rzN
��P"�N��NNH	�OW�xN�O
�7N�:`N��HN��=Nx  �  j  r  �  �  �  
�  �    5  f  �  �  i  h  �  y  h  �  �  �   �    �  n  $  �      S  �  �  �  �  �  �  2  
  V  c  �  �    #  �  �  �  �  V  �  �  z  i  �    \    �  K  �  j  =  �  �    �    �  �  4  	�  �  �  S  �  �<�9X<e`B<t�;��
;o�o����e`B�T���D���49X�49X�49X����T���D���T����t��u��o���
�u�ě���C����㼋C�����ě���1���㼓t���t����
��1��1��9X�ě����ͼ�����/��/��/�+�o�o�\)�\)�t��\)����P��w�#�
�',1�0 Ž49X�<j�@��0 Ž0 Ž49X�e`B�H�9�L�ͽT���Y��ixս�+��7L��^5��t����
��1��^5�ě� ),)(!)-+)$""#'/;:80/."""""""""�����

����

<HUantroda]U<,$'����� ��������������������������~}��������������������ghst������tqhfgggggg������������������������������������������



��������������������������������DHIU]acegfaUOHBBDDDD_anxxz�����znga\YXZ_��������� ������������������	���������`hjt��������ztih````��������������������569BFORPOBB655555555fhtttxxthfffffffffff����������������������� ��������hhtt|������{thhhhhhh$6COZXSE@861*ou�������������yuooo(5B[cb[Z\WSNB5)��
#/<HE</#
��!#//0<<<9/##!!!!!!!!_agmnz~|zna________nnuz��������zqnnnnn)6BKPRUSOB62%#)*5BBB>=5))########����������������������������������������mz�����������ztmhcdm������������������������������������������

�������S[ht��������tqa[UPPS��������������������JN[gt�����{tg[NLJJJJYajnz��zzna\YYYYYYYY��������������������RYgmz����������maYRR������������������������������������������������������������V[]ght~zvttg][VVVVVggmt����tggggggggggg��������������������[bnotzncbZ[[[[[[[[[[��������������������eht����thfeeeeeeeeee����������������������������������������#$0<?EFGFF<0,!029<=IU]`bdc]UI<10/0u����������������qpu����������������������������������NOZ[bhrnhc[YOMNNNNNN�����������������������������������������������������������������������������������������������8<DHU`\UHC<888888888)5BN[_\N5)��
#(,/1/-#
������������������������FHUahjeaUPJHFFFFFFFF���
������������������������������������������������źֺʺɺƺɺҺֺߺ������ֺֺֺֺֺ�������������������������	� �������������������������������������������������������ؾʾ����������žʾ׾��������𼘼��������������������������������������;�,�*�)�&��"�;�H�a�u�z�}�~�z�m�a�T�H�;�� ��������'�4�O�^�d�c�Y�S�@�4�'�������������(�0�5�8�9�5�/�(����������������ÿĿſѿԿۿѿĿ������������<�;�/�.�)�/�<�H�U�X�W�U�I�H�<�<�<�<�<�<ĿĺĽĿ������������ĿĿĿĿĿĿĿĿĿĿ�.�(���(�/�4�A�M�Z�f�s�v�u�~�s�Z�M�A�.���ݽܽݽ��������������������O�B�6�-�)�&�)�0�6�B�O�S�[�k�w�t�o�h�[�O�N�B�A�9�A�H�N�Z�g�s�~�x�s�n�g�Z�N�N�N�N��������������������������������������������������������������������������������ìãàÖÒÎÈÁÇÉÓàìùú������ùì���y�x�s�o�t�x����������������������������������������*�6�C�L�N�C�@�6�*�����a�]�U�R�U�\�a�n�o�n�n�b�a�a�a�a�a�a�a�a�׾Ծʾʾʾ׾����׾׾׾׾׾׾׾׾׾��I�0�#���#�0�I�U�b�n�{ņŌŇł�{�n�b�I�{�z�x�{�|ŇŎŔŠŭŰůŭŪŦŠŔŇ�{�{��������¿·¼¿�����������������������ؿ.�"�	����پ׾�����"�.�3�;�B�G�M�G�.�������������������ʾ־޾����߾׾ʾ��������������������*�6�C�J�M�I�C�6������"�'�*�6�B�S�W�Z�e�[�Y�O�B�6�)��H�D�<�;�<�F�H�K�U�Z�U�U�H�H�H�H�H�H�H�H����� �������$������������������������������������������������޾s�f�Z�B�M�V�b�s���������������������s�����������������������������������������G�F�;�G�T�`�h�a�`�T�G�G�G�G�G�G�G�G�G�G����������������������������������������ƽƺ��ƿ����������������������������ƽ����۾ݾ�����	���	����������������������"�������������������������$�0�;�4�0�$�"�������������|�}�}�����������������������������������������������������������������������������������	�������
�	�������������������������������������������FF FFFF$F%F'F$FFFFFFFFFFF���������������������	��"�'�)�$���	���;�9�;�A�H�T�a�d�a�[�T�H�;�;�;�;�;�;�;�;�����������������������������������������O�I�H�M�K�O�[�[�h�m�t�x�t�h�\�[�O�O�O�OĳĪĦĚĚēĚĦĳĿ����������Ŀĳĳĳĳ����������������������������	����)�5�A�K�@�5�2�)�"���N�I�L�N�Z�g�p�g�f�Z�N�N�N�N�N�N�N�N�N�N�
����ۿҿؿݿ������ �(�*�+�(��
������'�1�1�)�'�����������V�R�I�H�E�=�I�V�b�o�{�~ǈǈǆ�{�u�o�b�V��������$�+�%�$������������������������������ĽнݽԽнĽ��������A�4��������$�4�A�M�W�a�d�Z�W�M�A�����r�L�E�L�Y�������ֺ������ֺ������S�I�F�:�-�$�-�:�F�S�_�`�_�X�S�S�S�S�S�S��	������������"�.�3�;�B�;�.�"��ù¹��������ùϹڹܹ߹ܹӹϹùùùùù����
�
�	�
��#�$�,�#�#�������������ݿӿѿĿ��������Ŀѿݿ�������'��'�4�7�@�M�R�M�C�@�4�'�'�'�'�'�'�'�'�f�[��������������������ּ���f�ֺպɺ����ɺɺֺغ�������׺ֺֺֺ�E�E�E�E~EzE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������������������������������������E*E(E#E$E'E*E7E>ECEHEPEWEVEPENECE7E7E*E*�a�g�a�`�T�H�;�/�/�&�#�/�;�B�H�M�T�a�a�a�������������ùǹ˹ù���������������������������!�$�.�9�.�*�!����������������(�5�A�B�A�<�5�(�%�������������������������������������������� * I R V 5 ; #  =   b U h J ; L C ^ - b G P @ 3 u x Q = G + U & S B < c : b P c " 3 f Q Z 4 ` w H F ^ E S g 5 ' ' 2 9 O y z & ; t m n 1 o � 9   3 2 < ?  {  �  h  �    �  X  �  ~  �  5  �   �  �  �  �  �  �  C  �  ,  
  �  2  �  ^  �  �  �  +  p  �  @  n  I     �  �  L  (  �  j  :  �  	  �  \  �  �  �  [  q  R  �  b  d  p  d  �  �  �  �  t  p  �  K  -  �  }  �    %  �  �     =  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  v  �    n  Y  A    �  �  �  @  �  �  x  2  �  �  j  ?    j  _  S  G  =  4  +  !      �  �  �  �  �  �  �  �  �  �  r  ^  K  5      �  �  �  �  �  l  P  2    �  �  �  �  r  �  �  �  �  �  �  �  j  P  6      �  �  �  �  �  �  i  K  �  �  �  �  �  �  �  �    {  w  s  n  g  a  Z  H  2      �  �  �  �  �  �  �  �  {  h  O  9  $      �  �  �  �  {  �  �  	#  	�  
  
O  
�  
�  
�  
�  
�  
�  
�  
"  	�  	�  �        `  i  r  z  �  �  �  y  j  V  ?  &    �  �  �  �  `  \  8  �  �  �  �  �  �      �  �  �  �  �  �  `  *  �  �  [       (  0  2  5  1  +       �  �  �  �  �  r  R  1    �  �  f  f  e  e  d  d  c  c  b  b  k    �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  f  J  -  �  �  �  �  ~  r  d  V  H  :  +       �   �   �   �   �   �   �  3  H  Y  c  h  h  a  Q  5    �  �  �  k  3  �  K  �  �  �  g  h  e  _  Y  R  I  >  0      �  �  �  c  2  �  �  c   �  �  �  �  �  �  �  |  j  W  C  *  $  !    �  �  �  �  �  f  y  r  k  d  Z  P  E  <  4  ,  "         �  �  �  �  r  L  =  N  ]  h  g  _  O  @  ,    4    �  �  a  *  �  �      �  �  �  u  b  K  /    �  �  �  �  u  H    �  ]  �  j  �  |  �  }  t  h  a  [  G  .    �  �  �  �  f  G  -    �  �    5  M  c  u  �  �  �  v  J    �  h    �  ^  �  �  1   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  
  
  �  �  �  �  �  u  O     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  l  W  B  )      �  R  X  ]  b  g  i  k  m  t  �  �  �  �  �  �  �  �  �  �  �  $         �  �  �  �  �  �  �  �  {  ^  @     �  �  �  L  �  �  �  �  �  �  �  �  s  c  R  @  +    �  �  �  z  2   �  �  �  �                �  �  �  a  -  %  �  �  �  �  �        �  �  �  �  |  V  3    �  �  �  l  G    �  N  ?  D  I  N  R  M  I  D  ?  8  1  *  "      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  Z  �  �  �  �  �  �  �  �  |  i  V  C  /      �  �  �  �  r  �  �  �  �  w  g  W  F  2    �  �  �  �  r  G    �  y    �  �  �  �  �  �  |  p  d  X  L  ?  2  %      
    �  �  �  �  �  �  x  k  ^  Q  C  2       �  �  �  �    L     �  �  �  �  �  �  �  �  �  z  r  k  d  ]  V  P  I  B  ;  4  -  2  %    
  �  �  �  �  �  �  �  �  �  |  m  _  V  O  G  @  
       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  K  A  7  ,        �  �  �  �  �  �  �  p  Z  D  .    c  P  <  *      �  �  �  �  �  |  f  P  (  �  �  �  �  �  �  �  �  s  T  5    �  �  �  �  �  �  }  H    �    4  �  �  �  �  |  w  q  g  ]  S  I  ;  )      �  �  �  �  �  �        �  �  �  �  �  �  �  �  |  h  S  =  &    �  �  �  #    �  �  �  �  �  �  u  k  d  ]  X  S  P  M  K  I  G  F  �  n  P  6      �  �  �  �  �  �  �  w  g  W  F  7  '    �  �  �  �  �  �  �  �  y  �  �  �  �  s  L    �  �  �  �  �  �  �  �      /  -         �  �  �  �  �  �  X  "  �  �  �  �  �  �  �  �  �  }  y  ~  �  �  �  �  �  &  j  �  �  V  B  .      �  �  �  �  �  �  �  n  W  ?  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  <  �  �  �  G  �  �  Y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  v  p  i  c  a  j  s  z  y  u  n  c  U  >  #    �  �  }  ;  �  �  8   d  i  e  b  _  [  X  U  Q  N  K  I  G  F  E  D  B  A  @  ?  =  �  ~  |  z  y  z  w  q  d  U  D  0    �  �  �  `      &  �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  |  o  n  r  K  Y  K  2    �  �  �  �  m  G     �  �  �  h    �  �  L  �  �  �  �        �  �  �  �  �  �  �  �  e  1  �  �  �  |  �  �  �  �  �  �  �  �  �  �  `  2  �  �  �  �  �    z  �  8  H  J  D  7  '    �  �  �  �  z  F  �  m  �     E  #  �  }  D  �  �  g  O  f  �          �  w  �  a  �    �  j  _  U  J  ?  3  #      �  �  �  �  �  �  w  b  L  7  !  =  &    �  �  �  �  �  �  �  o  i  n  K     �  �  +  �  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  6  
  �  �  �  �  �  �  �  �  �  �  �  y  l  `  T  H  ;  *       �   �   �      �  �  �  �  �  �  �  e  G  &    �  �  �  �  �  t  L  �  {  q  h  ^  U  K  H  H  G  G  G  F  D  ?  9  3  -  (  "      �  �  �  �  �  T    �  �  �  d  &  �  u  
  �  �  X  �  �  �  j  O  4       �  �  �  �  �  |  s  u  w  v  s  p  �  �  |  l  \  L  <  ,    	  �  �  �  �  �  �  w  ]  B  (  4  %    �  �  �  �  e  7    �  �  w  J    �  �  �  a    �  0  \    �  	)  	v  	�  	�  	�  	�  	]  	  �  L  �  �  u  +  +  �  �  w  g  Z  I  4      �  �  �  x  E    �  c  �  P  �  �  �  �  u  f  W  G  6  &      �  �  �  �  �  {  a  F  +  S  K  B  7  ,      �  �  �  �  �  b  ?    �  �  �  �  a  �  �  j  K  #  �  �  �  p  :  �  �  x  2  �  �  \    �  �  �  b  ,    �  �  �  u  P  +    �  �  �  n  H  =  P  i  �