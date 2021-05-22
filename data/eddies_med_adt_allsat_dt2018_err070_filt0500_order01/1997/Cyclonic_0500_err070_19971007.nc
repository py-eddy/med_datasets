CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��G�z�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��S�   max       <u        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>fffffg   max       @F�z�G�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33332    max       @vtz�G�     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�   max       ;D��        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�9�   max       B4��        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�nJ   max       B4��        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?^Z�   max       C���        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ??��   max       C��{        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       Pp        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�R�=   max       ?��M:�        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��S�   max       <u        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>fffffg   max       @F������     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vtz�G�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O@           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D&   max         D&        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?��u%F        W�                     ;            
      K   %                        #   *      6   7      "   '   i         c   )         C         /         O      C   	                     	   $   %   "                        N���N� �O"N�?�No��O�P$O��JNT�CO ��N���N�.P���Py��N?��O �OeܛOU��P^@�OI5OdJ�O��WOŞM��	P��PgqO3ɰP
�=O���P�O%�cN?�tPX��Pt�N�fO��P4jsNB�)O!O��N�p'N�)RP0��N�(�P�qNxQ�OPl�O��O"%�N/V%Noh8Oee�N���O�X�P�OL��ON�O�
M���O5NpхN�P2N4�NF
�<u:�o�o�o��o���
���
�t��T���e`B�e`B�u�u�u��o��C���t����
��9X��9X���ͼ�����/��h���������o�o�\)�\)�\)�\)�t�����w��w�#�
�49X�D���L�ͽT���aG��e`B�q���q���q���}󶽃o��������7L��7L��hs��t����P������Q콼j��j�ȴ9������S����������������#/2145/)#?BN[dgkljgc[VNDB:<??
#&%%%#

�qz���������zvrqqqqqq������������������)BO[gt}}zt[NB*��������������������66:CENOWSSQOEC666666`hot���������|tjh_``�������������������������������������������#<Tn{����{b<
��������������������tv�������������������������������������������������������������)6BFOQRRQOB6)!�����������������}|�����!((#���������������������������x{����������������zx��
#/)&#
��������~��������}~~~~~~~~~~���
#.HU^eh`<#�����#/<FKW^]UH<2#
//;AEHJMOQOH;/)'&'*/T[mz����������maTPNTz������������zpoqqtz���������������(5BN[t�����tg\NB5)%(���������������������������	 !���������
+<[[D<#
���������������������������489;BO[]mqsr[OB<:544?E[t���������g[JC77?��������������������������	�������������� ���������3;HRSNHH;70033333333�����������������������/HLUZVQMF#
������	��������0<IUityxnbUI;0#
Y[`gpspkgg^[WVYYYYYY�����������������'+,(�����_amnz��������zniaVV_MNR[`a_[VNNMMMMMMMMM�������������������������������������������������������������������'!��������t~�����������tmlnntABFOQ[hu}�}tkh[OB<>A_ght����������tig^[_nz�������������zvrpn#%')#$)5BN[\[QND:5)fgpt|�������thg`fffft�������vtqtttttttttz�����������|uzzzzzz+/<<CB?<5/.(++++++++�w�t�s�t¦«¬¦¢�<�6�/�%�)�/�<�H�U�X�U�Q�H�>�<�<�<�<�<�<���������������Ŀѿܿݿ�ݿۿѿϿĿ������I�@�E�E�I�V�b�o�v�{��{�o�n�b�V�I�I�I�I��������������������������������������Z�T�P�R�Z�[�g�s�����������z�s�g�Z�Z�Z�Z�g�^�A�5����������5�A�N�r�}�����s�g�M�E�1�(�+�5�A�Z�f�s�����������s�f�Z�M���׾˾ʾ������ʾ׾����������ݽٽн̽Ľƽнݽ������������ݽݾ����������������������þ����������������ܻԻлȻ˻лܻ�������ܻܻܻܻܻܻܻ��������~�u�o�g�S�T�g�������������	�	��ݿ��y�`�I�A�?�G�m���Ŀݿ����5�B�A�(���/�(�#�/�;�H�T�Z�T�H�;�;�/�/�/�/�/�/�/�/�/�%�#���
�	�
��#�/�<�H�N�U�_�U�H�<�/�a�m�v�z�m�`�T�@�;�.�)�'�#� �+�.�;�G�T�a�s�n�k�n�q�s�{�������������������������s������Şŝŭ��������*�C�S�`�c�O�C�*���ŠşŕŔŒŐŔŠŭŹ������������ŹŭťŠ�ݿѿĿ��������������������Ŀѿؿ�����������������$�0�=�D�K�I�8�3�)�$��E*E"E E E"E*E7ECEIEPESETEPENECE7E*E*E*E*�Y�W�T�Y�Y�b�f�j�k�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y����������������)�6�B�]�P�<�;�)���y�g�o��������������	����������������y��������������������������������������������������������"�/�A�A�I�N�U�Q�H���èãâì�������������!������ùè� �)�-�B�O�X�hāĚĳĿ������Ŀ�t�R�B�6� �������������	�������
��
�	��ƳƩƭƳƹ��������������ƳƳƳƳƳƳƳƳ�r�f�G�3�.�/�4�@�f�������Ӽ޼ּݼؼ����r�������ú���-�:�l�v�k�o�S�:�!���ɺ��e�b�e�r�y�~�������������~�r�e�e�e�e�e�e���ʾ�������������	�"�2�1�;�.�"��	����	�������/�H�T�m�w�u�v���{�m�a�;�/�Ň�~�{�w�{ŇŔşŠťŠŔŇŇŇŇŇŇŇŇ��������ܾ����	��"�.�/�1�.�+�"��(�������(�5�A�N�V�Z�]�Z�N�I�A�5�(������(�5�5�8�5�(���������Ň�~ŇŏŔŠŭŹ��������ŹŭŠŔŇŇŇŇ�ϿĿ������������Ŀݿ����?�?�2����Ͽ��������y���������������������������������������������н��7�@�:�'���ؽʽ�����������������)�*�)������������������
�������!�+�:�J�S�`�e�`�U�G�4�!��������n�y�{�������Ľݽ���!����нĽ������������������������ʾӾξϾʾǾ������������������������������������������$������$�$�0�2�=�A�=�0�$�$�$�$�$�$�&����"�4�B�M�]�r���������r�f�Y�@�&�m�a�a�T�N�R�T�_�a�c�m�z����z�t�m�m�m�m�:�-���!�F�_�k�h�l�x������������l�S�:�|�g�[�N�B�9�B�[¿������������¿²�|�"� �� �������	��"�/�6�;�>�B�C�@�;�/�"�6�+�*�����*�0�6�C�O�\�_�`�\�[�O�C�6�ܹйιй۹�����'�3�@�J�G�6�������E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E��z�t�n�j�d�_�\�Z�a�b�n�rÇÓ×ÛÙÓÇ�z�H�G�<�7�/�,�/�7�<�H�I�O�N�U�V�U�H�H�H�H²«­²¿����������¿²²²²²²²²²����������������������������������������D{DxD{D{D�D�D�D�D�D�D�D�D{D{D{D{D{D{D{D{ G D " = W 0 4 7 c 5 # L a q } L C ( M 3 Y R ' W X Q Z J P y � 7 ' h _ O  P ` 6 6 ` 2 = a M @ P . h p b & ^ h .  S ` j h , k F  �  �  9  �  �  ?  �  P  �  !  �  �  �  �  �  Z  �  �  �  �  �  �  C  4  (  �  �  �  �  �    _  �  y  �  �  �  Q  �  �  �  �    �    �  �  x  \  Q  �  2  �  I  #  �  �  x  '  �  �  �  R  p;D���#�
�e`B�e`B��`B���㽃o��/��C����
��j��1��Q�P�`���
���t��49X�D���,1�,1�y�#��C��+��1��{�8Q콅���hs�I��D���'1'�����'�+����49X�aG���^5�m�h�e`B�����P���#��7L���T���T���-��O߽�\)��{�������`��/�����j��G��Ƨ��l�������xս����B�B��Bu�B&�BoVB�B�B��B0��B��B4��B J�B%�B*�}A�9�B{�B!�B�~BO4B��BTB��B�5B+�B�?B]�A��mB �B f�BA,B	:�BB� B$ \B �B��B	�B�B�(B�A�DBM\BhaB��B&��B	#BV|B�B�B��B+�B*��Bj&B�?B
��BO�B
ZB2B�mBb�B
*?B߲B
�FB@B��B��BO�B��B{EB��B��B��B0ŉB�0B4��B U�B&�B+4�A�nJBB�B!A�B��BG�B�B�B<�BſB/�BG:B��A���A�r]B �VB@�B	FB=EBD�B$,3B!8�B��B	@lB=�B��B��A��%BAEBCwB��B'@rB	3�BKeBB�B@FB	�BA�B*�mB@zB�MB
?JBF�B
�B��B��B�SB
?RB��B
C�B�A��zA��@Aw�*B	�A�%'A���A��!A>�ASfA+ҶAK��@�q�A� �Av�(A���Aµ�Ae!^A��9A�y�A���Aw)�B	r�C��@��eA��A�X#A���A��9A�^QA���AZ�B�o@�z�@_u�@#�AW�<A���A� =A[>A�ԭA���A��!A�wxAr�{A,\jA�Q�ArQA'�#AMj�A��(B
�@��A��@��@A�=A�_B ��?^Z�C���A�k�A��]A�v�A�
�C���A���A�ELAw�B�A�j�A�qA��CA=/�AS��A+hAK�|@�BWA���AmۭA���A�Af�dA�	A�[ZA�~gAuCB	� C��I@���A�p�A���A��oA�~�A�vA�t�A[�B��@�e~@c��@6AX�A��A�WA\��A���A���A�{A�p�Ar��A-	_A�m�A��A)��AL��A�[�B	D�@�CA�|�@�	`A��A��WB �!??��C��{A�Y�Aâ_A���A��C���                     <            
      L   %                        $   *      7   8      #   (   j         c   )         D         0         O      C   	                     	   $   &   #                                              )                  ?   A               3         #         /   )      '      /         3   ?      !   )                  +      /         +                  )   )                                                                  5   ?               3         #            '      '               /   =      !   )                  %      )         +                  )   %                           N���N(��N��N�?�No��O�O���OW?/NT�CO ��N\z�NU�]PpPc�jN?��N�u�OeܛOK<~P^@�O��OdJ�O��WN�)tM��	O�ؗO��O"�P
�=O;@qO�)qN���N?�tPA}rPi��N�fO��P,��NB�)O!O��N�p'N�)RO�m:N�(�O�ڪNxQ�OC��O��O�:N/V%Noh8Oee�N���O�X�O�e�OL��ON�Of�
M���O5N0�N�P2N4�NF
�  �  �  �  7    f  8  5  �  �  *  �  �  (     I    H  7  �  �  �  Q  8  G  �  �  �  �    �  �  �  �  �  %  W  �  G  �  �  [  	$  ^  n  �  c  �  �    N  
  �  M  T  �  �  D  �  U  &  �  �  �<u�o��o�o��o���
���ͼT���T���e`B��C���o������C���o���㼓t���1��9X���ͼ��ͼ����\)��h�H�9��P�o���<j��o��w�\)�0 Žt��t����'�w�#�
�49X�D���L�ͽ�O߽aG��}�q���u�q����%��o��������7L��7L���P��t����P��{��Q콼j��vɽȴ9������S���������������� #/////12/,#      ABN[aghjgg[NIB=>AAAA
#&%%%#

�qz���������zvrqqqqqq������������������")5B[gknmf[NB5)%"��������������������66:CENOWSSQOEC666666`hot���������|tjh_``��������������������������������������������
#Un{��{b<0#���������������������{�������������������������������������������������������������)6BOPQRQOKB6)!�����������������}|���%% ��������������������������x{����������������zx��
##%##
�������~��������}~~~~~~~~~~�
 /<FNK</# ����#/;CGQWUH</#+/2;?DHJLNOMH;/*('(+T[mz����������maTPNTz~������������zxxwwz������
�������O[gtx���ytgf[OOOOOOO����������������������������������
*<XYB<#
����������������������������489;BO[]mqsr[OB<:544@G[t���������g[LB88@��������������������������	�������������� ���������3;HRSNHH;70033333333����������������������
<AIJD</#
 �������	��������#0<IUeqwwvnbUI=50#Y[`gpspkgg^[WVYYYYYY�����������������'+,(�����`ansz~������znmaXW``MNR[`a_[VNNMMMMMMMMM�������������������������������������������������������������������'!��������u}������������tnmopuABFOQ[hu}�}tkh[OB<>A_ght����������tig^[_xz�������������yvttx#%')#$)5BN[\[QND:5)qt}�������tjqqqqqqqqt�������vtqtttttttttz�����������|uzzzzzz+/<<CB?<5/.(++++++++�w�t�s�t¦«¬¦¢�/�-�/�/�0�<�=�H�U�V�U�O�H�<�/�/�/�/�/�/���������������Ŀѿҿٿֿѿ˿Ŀ����������I�@�E�E�I�V�b�o�v�{��{�o�n�b�V�I�I�I�I��������������������������������������Z�T�P�R�Z�[�g�s�����������z�s�g�Z�Z�Z�Z�5�(����"�(�5�A�N�_�l�t�s�k�g�Z�N�A�5�M�G�A�4�2�.�0�4�<�A�M�Z�f�s������s�W�M���׾˾ʾ������ʾ׾����������ݽٽн̽Ľƽнݽ������������ݽݾ����������������������������������������ֻܻлʻ̻лܻ�����ܻܻܻܻܻܻܻ��������������|�t�x��������������������ݿѿ��y�`�L�C�A�T�m�����Ŀݿ�޿��2�/���/�(�#�/�;�H�T�Z�T�H�;�;�/�/�/�/�/�/�/�/�/�(�#�����#�/�<�H�K�U�V�U�H�D�<�/�/�a�m�v�z�m�`�T�@�;�.�)�'�#� �+�.�;�G�T�a�s�o�l�n�r�}���������������������������s������Şŝŭ��������*�C�S�`�c�O�C�*���ŠřŕŔŔœŔŠŭŹ����������ŹŹŭŠŠ�ݿѿĿ��������������������Ŀѿؿ�����������������$�0�=�D�K�I�8�3�)�$��E*E&E#E#E*E*E7ECECEPEQEREPEIECE7E*E*E*E*�Y�W�T�Y�Y�b�f�j�k�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y����������� �)�6�B�P�J�G�B�6�)��������r�w����������������������������������������������������������������������������������������"�/�A�A�I�N�U�Q�H�����ùððù���������������������������h�[�]�c�h�oāčĚĦĳĿ������ļĳĚč�h������������	�������	������������ƳƩƭƳƹ��������������ƳƳƳƳƳƳƳƳ�r�f�I�5�0�2�@�M�f�������ȼӼѼԼм����r�����ĺ���-�:�l�s�j�n�S�:�!���⺿���e�b�e�r�y�~�������������~�r�e�e�e�e�e�e���ʾ�������������	�"�2�1�;�.�"��	����	�������/�H�T�a�m�u�u��z�m�[�;�/�Ň�~�{�w�{ŇŔşŠťŠŔŇŇŇŇŇŇŇŇ��������ܾ����	��"�.�/�1�.�+�"��(�������(�5�A�N�V�Z�]�Z�N�I�A�5�(������(�5�5�8�5�(���������Ň�~ŇŏŔŠŭŹ��������ŹŭŠŔŇŇŇŇ�ѿƿĿĿ˿ѿ�����*�5�5�*������ݿѿ��������y�������������������������������������������н���-�;�5�!���ݽҽ�����������������)�*�)�����������������!���� ����!�*�:�H�S�]�S�R�G�:�.�!�������n�y�{�������Ľݽ���!����нĽ����������������������ʾϾ˾˾ʾƾ��������������������������������������������$������$�$�0�2�=�A�=�0�$�$�$�$�$�$�&����"�4�B�M�]�r���������r�f�Y�@�&�m�a�a�T�N�R�T�_�a�c�m�z����z�t�m�m�m�m�:�-���!�F�_�k�h�l�x������������l�S�:�t�[�U�O�[�t¦¿������������¿²�"� �� �������	��"�/�6�;�>�B�C�@�;�/�"�6�+�*�����*�0�6�C�O�\�_�`�\�[�O�C�6�ܹٹչֹܹ������'�3�7�>�,�������E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E��z�t�n�j�d�_�\�Z�a�b�n�rÇÓ×ÛÙÓÇ�z�<�9�/�-�/�9�<�E�H�N�L�H�<�<�<�<�<�<�<�<²«­²¿����������¿²²²²²²²²²����������������������������������������D{DxD{D{D�D�D�D�D�D�D�D�D{D{D{D{D{D{D{D{ G L " = W 0 % 1 c 5  V X d } M C ' M ( Y R " W d K W J N M d 7 $ f _ O  P ` 6 6 `  = a M 8 P 3 h p b & ^ _ .  M ` j ? , k F  �  S  �  �  �  ?  u  �  �  !  `  n  W  (  �    �  �  �  A  �  �  �  4  P  "  �  �  �  {  �  _  B  >  �  �  �  Q  �  �  �  �    �  �  �  �  x  7  Q  �  2  �  I  ^  �  �  �  '  �  E  �  R  p  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  D&  �  �  �  �  �  �  �  s  Z  9    �  �  a    �  y  &  �  �  k  q  v  }  �  �  �  |  y  m  Q  9  !  	  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  h  P  5    �  �  7  $      �  �  �  �  {  G    �  �  �  �  �  w  J    �         �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  h  T  f  _  V  L  B  6  )      �  �  �  s  A    �  �  o  8     Z  �  �    "  0  7  7  0      �  �  �  6  �  c  �  $  V  -  /  3  5  5  4  -  !    �  �  �  �  �  ^  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    r  e  X  �  �  �  �  �  �  �  �  �  }  q  b  T  ?     �   �   �   �   b          !  %  )  (  &  !      �  �  �  �  �  �  l  Q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  8  �  �  �  �  �  �  |  T    �  �  )  �  �  P    �  H  �  U    '        �  �  �  �  �  �  �  Y    �  n    �  �  k         	    �  �  �  �  �  �       2  E  P  X  `  h  p  5  C  F  7       �  �  �  s  I  !  �  �  �  �  Y    �  C    �  �  �  �  �  �  �  �  �  �  p  _  d    j  N  (  �  �  H  H  D  =  2  !  	  �  �  �  ~  U  )  �  �  |  ,  �  Q  �  7    �  �  �  �  �  �  �  �  �  �  �  `  -  �  �  X  �  �  �  �  �  �  �  �  �  �  }  Z  4    �  �  �  i    �  5  �  �  �  �  �  �  �  �  �  �  �  �  �  b  *  �  �  3  �  W   �  �  �  �  _  :    �  �  u  4  �  �  S  �  �  N  ,  �  e  �  
�    8  K  O  >    
�  
�  
p  
#  	�  	R  �  .  �  �  :  t  \  8  =  B  F  K  P  R  T  W  Y  _  h  r  {  �  b  2    �  �  �  �  �  �  �  "  B  G  <  #  �  �  �  s  \    �  m  �  �  r  �  �  �  �  �  x  P    �  �  �  5  �  �  (  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  O  !  �  �  q  -  �  �  �  �  �  �  �  �  �  �  �  q  K    �  �  *  �  \  �    1  X  v  �  �  �  �  �  �  �  F  �  �  r  ;    �  {  �  V  �  F  [  A  �  �    �  �  �  a    �    y  
�  	'  �  �  h  �  �  }  v  �  �  �  �  �  �  �  h  C    �  �  U    �  �  �  k  U  @  (    �  �  �  �  �  �  ~  o  g  ^  V  T  R  P  �  �  �  �  �  N    
�  
T  	�  	e  �  -  �  3  �  �  �  p  �  �  �  �  �  �  �  �  �  �  �  p  N    �  �  i  :  �  �  �  �  �  �    k  X  D  0      �  �  �  �  �  �  �  �  �  �  %  	  �  �  �  �  �  �  �  �  �  �  k  N  +    �  �  /  �  O  Q  <    �  �  y  <  �  �  �  [    �  |  �  �  #  W  �  �  �  �  �  �  �  �  �  �  �  �  z  e  M  5        �  �  G  #  �  �  �  �  �  �  �  �  j  =    �  �  �  R     �  �  �  �    \  -    �  �  �  N    �  n    �  Z  �  ?  �  �  �  }  e  J  0    �  �  �  �  �  j  L  .    �  �  �  �  _  [  J  :  )         �  �  �  �  �  �    	            u  �  �  	  	$  	  �  �  �  `     �  �  E  �  -  b  X  �  M  ^  A    �  �  �  �  t  �  �  w  T  .    �  �  �  `  <  �  +  h  n  g  T  ?  ,  *  &    �  �  �  �  l    �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _    �  C  �  A  c  `  Y  I  3    �  �  �  �  e  =    �  �  B  �  I  �  �  �  r  Q  -    �  �  u  A    �  �  w  N  $  �  �  -   �  �  �  �  �  �  w  k  `  U  I  ;  ,      �  �  �  �  U      �  �  �  �  �  �  �  i  =    �  �  `    �  �  <  �  �  N  D  :  0  $    �  �  �  �  �  �  �  �  �  �  �  y  i  X  
    �  �  �  �  �  h  E    �  �  �  n  0  �  �  ~  H  �  �  �  �  �  �  �  �  �  �  w  k  ]  N  5    �  �  �  �  Z  M  #  �  �  �  \  O  +  �  �  �  n  L  o  P  �  �  �  r  j    S  Q  O  R  K  =  %    �  �  >  �  �    �  q    �  �  �  �  �  i  E    �  �  �  R    �  �  L  �  �  %  �    �  �  �  �  �  �  �  u  _  F  +    �  �  �  �  v  Q  
  �  C  7  <  @  D  B  9  +        �  �  �    J    �  k    a  �  �  �  �  M    �  �  k  .  �  �  x  ;  �  �  y  /  �  �  U  R  @  )  ,  2    �  �  �  �  A  �  �  j  +  �  �  F  �        !    �  �  �  �  �  q  Q  ,    �  �  �  V  !  �  �  �  �  o  M  '    �  �  �  h  3  �  �  S  �  �  6  �  f  �  �  �  �  �  �  �  }  t  j  \  I  7  $       9  R  k  �  �  �  �  �  M    �  �  �  k  A  !    �  �  |  L  !  �  �